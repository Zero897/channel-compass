from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import portable_path


RULE_VERSION = "prototype_dynamic_due_v1"
SCORE_VERSION = "health_score_v1"
SOURCE_LABEL = "AFFT企业提供脱敏模拟数据_原型规则与诊断"
CUSTOMER_SCOPE_LABEL = "当前有应收或近180天有采购或风险事件的客户"
CUSTOMER_DERIVED_COLUMNS = [
    "客户主键",
    "试点范围命中",
    "试点品牌代理",
    "试点区域",
    "营收质量分",
    "库存周转暴露分",
    "付款行为分",
    "信用暴露分",
    "合作稳定性分",
    "综合健康度",
    "健康度等级",
    "健康度证据",
    "健康度口径",
    "健康度版本",
    "高库存风险商品采购金额",
    "高库存风险SKU数",
    "近180天采购总额",
    "高库存风险商品采购占比",
]
STAGE_ORDER = {
    "数据待刷新": -1,
    "未临期": 0,
    "到期前5天": 1,
    "超期1-30天": 2,
    "超期31-60天": 3,
    "超期61-120天": 4,
    "超期120天以上": 5,
}


def _normalize_identifier(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def _stage(overdue_days: float, days_to_due: float | None) -> str:
    if overdue_days > 120:
        return "超期120天以上"
    if overdue_days > 60:
        return "超期61-120天"
    if overdue_days > 30:
        return "超期31-60天"
    if overdue_days > 0:
        return "超期1-30天"
    if days_to_due is not None and 0 <= days_to_due <= 5:
        return "到期前5天"
    return "未临期"


def _action(stage: str, performance: str) -> tuple[str, str]:
    actions = {
        "未临期": ("持续监测，到期前5天进入提醒队列", "常规监测"),
        "到期前5天": ("提醒客户付款并人工确认回款计划", "提醒"),
        "超期1-30天": ("电话及邮件催款，确认责任人与承诺回款日", "催收"),
        "超期31-60天": ("建议正式发函并升级财务或信用复核", "升级催收"),
        "超期61-120天": ("建议进入停止发货审批并专项催收", "停发审批"),
        "超期120天以上": ("建议进入法务及诉讼可行性复核", "法务复核"),
    }
    action, strength = actions[stage]
    if performance == "履约较差" and stage.startswith("超期"):
        return action + "；因历史履约较差，建议同步进入停止发货审批", "严格管控"
    return action, strength


def _suggested_owner_role(stage: str) -> str:
    return {
        "到期前5天": "客户经理",
        "超期1-30天": "客户经理/财务",
        "超期31-60天": "信用管理/财务",
        "超期61-120天": "业务负责人/信用管理",
        "超期120天以上": "法务/信用管理",
        "数据待刷新": "数据负责人",
    }.get(stage, "客户经理/信用管理")


def _rank_good(series: pd.Series) -> pd.Series:
    return series.astype(float).rank(pct=True, method="average").mul(100).fillna(50)


def _rank_bad(series: pd.Series) -> pd.Series:
    return 100 - _rank_good(series)


def _latest_customer_features(features_path: Path) -> pd.DataFrame:
    columns = [
        "customer_id",
        "order_date",
        "region",
        "gross_margin_ratio",
        "prior_order_count",
        "prior_sales_30d_growth",
        "prior_return_ratio_180d",
        "prior_overdue_payment_rate",
        "prior_overdue_amount_ratio",
        "prior_avg_payment_age_days",
        "prior_extension_count",
    ]
    frame = pd.read_csv(features_path, usecols=columns, dtype={"customer_id": "string"})
    frame["customer_id"] = _normalize_identifier(frame["customer_id"])
    return frame.sort_values(["customer_id", "order_date"]).drop_duplicates(
        "customer_id", keep="last"
    )


def _customer_product_exposure(
    company_dir: Path,
    processed_dir: Path,
    snapshot_date: pd.Timestamp,
    prototype_scope: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, set[str], pd.DataFrame]:
    inventory = pd.read_csv(
        processed_dir / "company_inventory_health.csv",
        dtype={"sku_id": "string"},
        low_memory=False,
    )
    inventory_snapshot_date = pd.Timestamp(inventory["snapshot_date"].max())
    inventory["_risk_rank"] = inventory["risk_level"].map(
        {"绿色": 0, "黄色": 1, "红色": 2}
    ).fillna(0)
    sku_risk = (
        inventory.sort_values(["sku_id", "_risk_rank", "inventory_value"])
        .groupby("sku_id", as_index=False)
        .agg(
            inventory_risk_rank=("_risk_rank", "max"),
            inventory_value=("inventory_value", "sum"),
            max_coverage_days=("coverage_days", "max"),
            max_inventory_age=("max_inventory_age", "max"),
            product_line=("product_line", "last"),
        )
    )
    sku_risk["库存风险等级"] = sku_risk["inventory_risk_rank"].map(
        {0: "绿色", 1: "黄色", 2: "红色"}
    )
    sales = pd.read_csv(
        company_dir / "销售流水.csv",
        usecols=["出库日期", "客户编号", "客户名称", "物料编码", "产品线名称", "客户大区", "数量", "销售金额_折扣后_含税"],
        dtype={"出库日期": "string", "客户编号": "string", "物料编码": "string"},
        low_memory=False,
    )
    sales["销售日期"] = pd.to_datetime(sales["出库日期"].str[:10], errors="coerce")
    sales["客户编号"] = _normalize_identifier(sales["客户编号"])
    scope_start = pd.Timestamp(str(prototype_scope.get("start_date", "1900-01-01")))
    scope_end = pd.Timestamp(str(prototype_scope.get("end_date", snapshot_date.date())))
    pilot_customers = set(
        sales.loc[
            (sales["销售日期"] >= scope_start)
            & (sales["销售日期"] <= scope_end)
            & (sales["产品线名称"] == prototype_scope.get("brand_proxy", ""))
            & (sales["客户大区"].isin(prototype_scope.get("regions", []))),
            "客户编号",
        ]
    )
    sales = sales[
        (sales["销售日期"] > snapshot_date - pd.Timedelta(days=180))
        & (sales["销售日期"] <= snapshot_date)
    ].copy()
    sales["物料编码"] = _normalize_identifier(sales["物料编码"])
    sales["采购金额"] = pd.to_numeric(
        sales["销售金额_折扣后_含税"], errors="coerce"
    ).fillna(0).clip(lower=0)
    sales["采购数量"] = pd.to_numeric(sales["数量"], errors="coerce").fillna(0).clip(lower=0)
    purchase_customers = (
        sales.sort_values("销售日期")
        .groupby("客户编号", as_index=False)
        .agg(客户名称=("客户名称", "last"), 近180天采购总额=("采购金额", "sum"))
    )
    purchases = (
        sales.groupby(["客户编号", "物料编码"], as_index=False)
        .agg(
            近180天采购金额=("采购金额", "sum"),
            近180天采购数量=("采购数量", "sum"),
            最近采购日期=("销售日期", "max"),
        )
    )
    purchases = purchases.merge(
        sku_risk,
        left_on="物料编码",
        right_on="sku_id",
        how="left",
    )
    purchases["库存风险等级"] = purchases["库存风险等级"].fillna("未匹配")
    total = purchases.groupby("客户编号")["近180天采购金额"].sum()
    risky = purchases[purchases["库存风险等级"].isin(["黄色", "红色"])].copy()
    risky["采购风险暴露金额"] = risky["近180天采购金额"]
    risky["采购风险暴露占比"] = risky.apply(
        lambda row: row["采购风险暴露金额"] / total.get(row["客户编号"], 0)
        if total.get(row["客户编号"], 0) > 0
        else 0,
        axis=1,
    )
    risky["采购暴露编号"] = risky.apply(
        lambda row: "EXP-"
        + hashlib.sha256(
            f"{snapshot_date.date()}|{row['客户编号']}|{row['物料编码']}".encode("utf-8")
        ).hexdigest()[:12].upper(),
        axis=1,
    )
    risky["公司端库存金额（快照）"] = risky["inventory_value"].fillna(0).round(2)
    risky["库存快照日期"] = inventory_snapshot_date.date().isoformat()
    risky["最大库存覆盖天数"] = risky["max_coverage_days"].fillna(0).round(2)
    risky["最大库龄天数"] = risky["max_inventory_age"].fillna(0).round(2)
    risky["产品线"] = risky["product_line"].fillna("未知")
    risky["对象口径"] = "客户采购暴露，不代表客户持有库存"
    risky["数据来源"] = SOURCE_LABEL
    exposure = risky[
        [
            "采购暴露编号",
            "客户编号",
            "物料编码",
            "产品线",
            "库存风险等级",
            "近180天采购金额",
            "近180天采购数量",
            "采购风险暴露金额",
            "采购风险暴露占比",
            "公司端库存金额（快照）",
            "库存快照日期",
            "最大库存覆盖天数",
            "最大库龄天数",
            "最近采购日期",
            "对象口径",
            "数据来源",
        ]
    ].sort_values(["库存风险等级", "采购风险暴露金额"], ascending=[True, False])
    summary = (
        exposure.groupby("客户编号", as_index=False)
        .agg(
            高库存风险商品采购金额=("采购风险暴露金额", "sum"),
            高库存风险SKU数=("物料编码", "nunique"),
        )
    )
    summary["近180天采购总额"] = summary["客户编号"].map(total).fillna(0)
    summary["高库存风险商品采购占比"] = (
        summary["高库存风险商品采购金额"]
        / summary["近180天采购总额"].replace(0, np.nan)
    ).fillna(0)
    return exposure, summary, pilot_customers, purchase_customers


def build_prototype_alignment(
    company_dir: Path,
    features_path: Path,
    processed_dir: Path,
    feishu_dir: Path,
    report_path: Path,
    prototype_scope: dict[str, object],
) -> dict[str, object]:
    customers = pd.read_csv(feishu_dir / "企业渠道客户.csv", dtype="string").fillna("")
    customers = customers.drop(columns=CUSTOMER_DERIVED_COLUMNS, errors="ignore")
    events = pd.read_csv(feishu_dir / "企业风险事件.csv", dtype="string").fillna("")
    tasks = pd.read_csv(feishu_dir / "企业处置任务.csv", dtype="string").fillna("")
    features = _latest_customer_features(features_path)
    snapshot_date = pd.Timestamp(customers["快照日期"].max())
    due_config = dict(prototype_scope.get("dynamic_due", {}))
    run_mode = str(due_config.get("run_mode", "historical_replay"))
    if run_mode not in {"historical_replay", "business_current"}:
        raise ValueError("dynamic_due.run_mode必须为historical_replay或business_current")
    run_date = pd.Timestamp(str(due_config.get("run_date", snapshot_date.date())))
    max_data_age_days = int(due_config.get("max_data_age_days", 3))
    data_age_days = max(0, int((run_date.normalize() - snapshot_date.normalize()).days))
    stale_for_business = run_mode == "business_current" and data_age_days > max_data_age_days
    calculation_date = snapshot_date if run_mode == "historical_replay" else run_date

    ar = pd.read_csv(
        company_dir / "应收快照_月末24期.csv",
        usecols=[
            "快照时间",
            "合同号",
            "客户编号",
            "客户名称",
            "项目名称",
            "销售订单号",
            "最终承诺还款日期",
            "是否展期",
            "超期天数",
            "应收金额",
            "超期应收金额",
        ],
        dtype="string",
        low_memory=False,
    ).fillna("")
    ar["快照时间"] = ar["快照时间"].str[:10]
    ar = ar[ar["快照时间"] == snapshot_date.date().isoformat()].copy()
    ar["客户编号"] = _normalize_identifier(ar["客户编号"])
    ar["销售订单号"] = _normalize_identifier(ar["销售订单号"])
    ar["到期日"] = pd.to_datetime(ar["最终承诺还款日期"].str[:10], errors="coerce")
    ar["超期天数值"] = pd.to_numeric(ar["超期天数"], errors="coerce").fillna(0)
    ar["应收金额值"] = pd.to_numeric(ar["应收金额"], errors="coerce").fillna(0)
    ar["超期应收金额值"] = pd.to_numeric(ar["超期应收金额"], errors="coerce").fillna(0)
    ar["距到期天数"] = (ar["到期日"] - calculation_date).dt.days
    if stale_for_business:
        ar["动态到期阶段"] = "数据待刷新"
    else:
        ar["动态到期阶段"] = ar.apply(
            lambda row: _stage(
                float(row["超期天数值"]),
                float(row["距到期天数"]) if pd.notna(row["距到期天数"]) else None,
            ),
            axis=1,
        )
    project_contracts = set(
        _normalize_identifier(
            pd.read_csv(
                company_dir / "增值合同签约明细.csv",
                usecols=["合同编号"],
                dtype="string",
            )["合同编号"]
        ).dropna()
    )
    ar["合同标准号"] = _normalize_identifier(ar["合同号"])
    ar["业务类型"] = np.where(
        ar["合同标准号"].isin(project_contracts) | (ar["项目名称"].str.strip() != ""),
        "企业级项目类",
        "常规消费品分销",
    )

    exposure, exposure_summary, pilot_customers, purchase_customers = _customer_product_exposure(
        company_dir, processed_dir, snapshot_date, prototype_scope
    )
    original_customer_ids = set(customers["客户编号"])
    original_orphan_exposure = exposure[~exposure["客户编号"].isin(original_customer_ids)].copy()
    event_customer_names = events[["客户编号", "客户名称"]].drop_duplicates("客户编号", keep="last")
    customer_names = dict(
        zip(purchase_customers["客户编号"].astype(str), purchase_customers["客户名称"].astype(str))
    )
    customer_names.update(
        dict(zip(event_customer_names["客户编号"].astype(str), event_customer_names["客户名称"].astype(str)))
    )
    master_customer_ids = (
        original_customer_ids
        | set(purchase_customers["客户编号"].astype(str))
        | set(events["客户编号"].astype(str))
    )
    missing_customer_ids = sorted(master_customer_ids - original_customer_ids)
    if missing_customer_ids:
        numeric_defaults = {
            "当前应收余额",
            "逾期应收金额",
            "逾期30天以上金额",
            "逾期60天以上金额",
            "最大逾期天数",
            "当前开放订单数",
            "预测高风险订单数",
            "预测高风险应收金额",
            "风险加权应收暴露",
            "最高模型概率",
            "最高风险分",
            "逾期占比",
        }
        missing_rows: list[dict[str, object]] = []
        purchase_ids = set(purchase_customers["客户编号"].astype(str))
        for customer_id in missing_customer_ids:
            row: dict[str, object] = {column: "" for column in customers.columns}
            row.update(
                {
                    "快照日期": snapshot_date.date().isoformat(),
                    "客户编号": customer_id,
                    "客户名称": customer_names.get(customer_id, "待核验客户名称"),
                    "风险等级": "绿色",
                    "风险来源": "近180天采购" if customer_id in purchase_ids else "风险事件补录",
                    "数据来源": SOURCE_LABEL,
                    "客户统计口径": CUSTOMER_SCOPE_LABEL,
                }
            )
            for column in numeric_defaults & set(customers.columns):
                row[column] = 0
            missing_rows.append(row)
        customers = pd.concat([customers, pd.DataFrame(missing_rows)], ignore_index=True)
    customers["客户统计口径"] = CUSTOMER_SCOPE_LABEL

    customer_numeric = customers.copy()
    for column in ["当前应收余额", "逾期应收金额", "逾期占比", "风险加权应收暴露"]:
        customer_numeric[column] = pd.to_numeric(customer_numeric[column], errors="coerce").fillna(0)
    customer_numeric = customer_numeric.merge(
        features.rename(columns={"customer_id": "客户编号"}), on="客户编号", how="left"
    )
    for column in [
        "gross_margin_ratio",
        "prior_order_count",
        "prior_sales_30d_growth",
        "prior_return_ratio_180d",
        "prior_overdue_payment_rate",
        "prior_overdue_amount_ratio",
        "prior_avg_payment_age_days",
        "prior_extension_count",
    ]:
        customer_numeric[column] = pd.to_numeric(
            customer_numeric[column], errors="coerce"
        ).fillna(0)

    customer_numeric = customer_numeric.merge(exposure_summary, on="客户编号", how="left")
    for column in ["高库存风险商品采购金额", "高库存风险SKU数", "近180天采购总额", "高库存风险商品采购占比"]:
        customer_numeric[column] = customer_numeric[column].fillna(0)

    customer_numeric["营收质量分"] = (
        0.4 * _rank_good(customer_numeric["prior_sales_30d_growth"].clip(-1, 2))
        + 0.3 * _rank_good(customer_numeric["gross_margin_ratio"])
        + 0.3 * _rank_bad(customer_numeric["prior_return_ratio_180d"])
    ).round(1)
    customer_numeric["库存周转暴露分"] = (
        100 * (1 - customer_numeric["高库存风险商品采购占比"].clip(0, 1))
    ).round(1)
    customer_numeric["付款行为分"] = (
        0.6 * (100 * (1 - customer_numeric["prior_overdue_payment_rate"].clip(0, 1)))
        + 0.4 * (100 * (1 - customer_numeric["prior_overdue_amount_ratio"].clip(0, 1)))
    ).round(1)
    exposure_ratio = (
        customer_numeric["风险加权应收暴露"]
        / customer_numeric["当前应收余额"].replace(0, np.nan)
    ).fillna(0)
    customer_numeric["信用暴露分"] = (
        0.7 * (100 * (1 - customer_numeric["逾期占比"].clip(0, 1)))
        + 0.3 * (100 * (1 - exposure_ratio.clip(0, 1)))
    ).round(1)
    customer_numeric["合作稳定性分"] = (
        0.4 * _rank_good(customer_numeric["prior_order_count"])
        + 0.3 * _rank_bad(customer_numeric["prior_sales_30d_growth"].abs())
        + 0.3 * _rank_bad(customer_numeric["prior_extension_count"])
    ).round(1)
    customer_numeric["综合健康度"] = (
        0.20 * customer_numeric["营收质量分"]
        + 0.20 * customer_numeric["库存周转暴露分"]
        + 0.25 * customer_numeric["付款行为分"]
        + 0.25 * customer_numeric["信用暴露分"]
        + 0.10 * customer_numeric["合作稳定性分"]
    ).round(1)
    customer_numeric["健康度等级"] = pd.cut(
        customer_numeric["综合健康度"],
        bins=[-np.inf, 60, 80, np.inf],
        labels=["红色", "黄色", "绿色"],
        right=False,
    ).astype("string")
    customer_numeric.loc[customer_numeric["风险等级"] == "红色", "健康度等级"] = "红色"
    customer_numeric["健康度证据"] = customer_numeric.apply(
        lambda row: (
            f"营收{row['营收质量分']:.1f}；库存暴露{row['库存周转暴露分']:.1f}；"
            f"付款{row['付款行为分']:.1f}；信用{row['信用暴露分']:.1f}；合作{row['合作稳定性分']:.1f}"
        ),
        axis=1,
    )
    customer_numeric["健康度口径"] = "五维透明诊断；库存维度为采购暴露，不代表客户持有库存"
    customer_numeric["健康度版本"] = SCORE_VERSION
    customer_numeric["客户主键"] = (
        customer_numeric["客户编号"] + "｜" + customer_numeric["客户名称"]
    )
    customer_numeric["试点范围命中"] = np.where(
        customer_numeric["客户编号"].isin(pilot_customers), "是", "否"
    )
    customer_numeric["试点品牌代理"] = str(prototype_scope.get("brand_proxy", ""))
    customer_numeric["试点区域"] = customer_numeric["region"].fillna("未知")

    performance = customer_numeric.set_index("客户编号").apply(
        lambda row: (
            "履约较差"
            if row["prior_overdue_payment_rate"] >= 0.20 or row["逾期占比"] >= 0.50
            else "需加强关注"
            if row["prior_overdue_payment_rate"] >= 0.05
            or row["逾期应收金额"] > 0
            or row["prior_extension_count"] > 0
            else "履约稳定"
        ),
        axis=1,
    ).to_dict()

    ar["客户履约等级"] = ar["客户编号"].map(performance).fillna("需人工复核")
    if stale_for_business:
        ar["动态建议动作"] = "停止实时分层，等待应收及回款数据刷新"
        ar["处置强度"] = "数据门禁"
    else:
        action_strength = ar.apply(
            lambda row: _action(row["动态到期阶段"], row["客户履约等级"]), axis=1
        )
        ar["动态建议动作"] = [value[0] for value in action_strength]
        ar["处置强度"] = [value[1] for value in action_strength]
    dynamic = (
        ar.groupby(
            ["客户编号", "客户名称", "业务类型", "动态到期阶段", "客户履约等级", "动态建议动作", "处置强度"],
            as_index=False,
        )
        .agg(
            订单数=("销售订单号", "nunique"),
            应收金额=("应收金额值", "sum"),
            超期应收金额=("超期应收金额值", "sum"),
            最早到期日=("到期日", "min"),
            最大超期天数=("超期天数值", "max"),
        )
    )
    dynamic["动态监控编号"] = dynamic.apply(
        lambda row: "DUE-"
        + hashlib.sha256(
            f"{snapshot_date.date()}|{row['客户编号']}|{row['业务类型']}|{row['动态到期阶段']}".encode("utf-8")
        ).hexdigest()[:12].upper(),
        axis=1,
    )
    dynamic["规则版本"] = RULE_VERSION
    dynamic["运行模式"] = run_mode
    dynamic["运行日期"] = run_date.date().isoformat()
    dynamic["计算基准日"] = calculation_date.date().isoformat()
    dynamic["数据更新时间"] = snapshot_date.date().isoformat()
    dynamic["数据新鲜度天数"] = data_age_days
    dynamic["数据状态"] = (
        "数据过期，已停止实时分层"
        if stale_for_business
        else "历史回放"
        if run_mode == "historical_replay"
        else "可用于业务时点分层"
    )
    dynamic["规则口径"] = (
        "历史回放：以应收快照日作为计算基准；动作仅供人工审批"
        if run_mode == "historical_replay"
        else "业务时点：仅在数据新鲜度门禁通过后分层；动作仅供人工审批"
    )
    dynamic["数据来源"] = SOURCE_LABEL
    dynamic = dynamic[
        [
            "动态监控编号",
            "客户编号",
            "客户名称",
            "业务类型",
            "动态到期阶段",
            "客户履约等级",
            "订单数",
            "应收金额",
            "超期应收金额",
            "最早到期日",
            "最大超期天数",
            "动态建议动作",
            "处置强度",
            "运行模式",
            "运行日期",
            "计算基准日",
            "数据更新时间",
            "数据新鲜度天数",
            "数据状态",
            "规则版本",
            "规则口径",
            "数据来源",
        ]
    ]
    worst = (
        dynamic.assign(_rank=dynamic["动态到期阶段"].map(STAGE_ORDER))
        .sort_values(["客户编号", "_rank", "超期应收金额"])
        .drop_duplicates("客户编号", keep="last")
        .set_index("客户编号")
    )

    for field in ["业务类型", "动态到期阶段", "客户履约等级", "动态建议动作", "处置强度", "规则版本"]:
        events[field] = events["客户编号"].map(worst[field]).fillna("")
    events.loc[events["模型版本"] == "business_rule", "建议动作"] = events.loc[
        events["模型版本"] == "business_rule", "动态建议动作"
    ]

    due_rows: list[dict[str, object]] = []
    due_candidates = dynamic[
        (dynamic["动态到期阶段"] == "到期前5天")
        & (dynamic["业务类型"] == "常规消费品分销")
    ]
    existing_event_ids = set(events["事件编号"])
    customer_by_id = customer_numeric.set_index("客户编号")
    for due in due_candidates.to_dict("records"):
        customer_id = str(due["客户编号"])
        event_id = f"AFFT-{snapshot_date.strftime('%Y%m%d')}-{customer_id}-DUE5"
        if event_id in existing_event_ids:
            continue
        customer = customer_by_id.loc[customer_id]
        row = {column: "" for column in events.columns}
        row.update(
            {
                "事件编号": event_id,
                "客户编号": customer_id,
                "客户名称": due["客户名称"],
                "风险类型": "到期前5天付款提醒",
                "风险等级": "黄色",
                "触发时间": snapshot_date.date().isoformat(),
                "数据快照": snapshot_date.date().isoformat(),
                "关键指标": (
                    f"{int(due['订单数'])}笔应收将在5天内到期；"
                    f"涉及金额¥{float(due['应收金额']):,.0f}；最早到期日{due['最早到期日']}"
                ),
                "风险金额": round(float(due["应收金额"]), 2),
                "置信说明": "应收快照到期日确定性规则命中，不是模型预测",
                "客户历史摘要": (
                    f"当前应收¥{float(customer['当前应收余额']):,.0f}，"
                    f"历史履约代理等级{due['客户履约等级']}"
                ),
                "规则解释": "实际读取最新应收快照的最终承诺还款日期，距离到期日不超过5天",
                "建议动作": due["动态建议动作"],
                "处理状态": "待处理",
                "模型版本": "business_rule",
                "data_source": SOURCE_LABEL,
                "业务类型": due["业务类型"],
                "动态到期阶段": due["动态到期阶段"],
                "客户履约等级": due["客户履约等级"],
                "动态建议动作": due["动态建议动作"],
                "处置强度": due["处置强度"],
                "规则版本": RULE_VERSION,
            }
        )
        due_rows.append(row)
    if due_rows:
        events = pd.concat([events, pd.DataFrame(due_rows)], ignore_index=True)

    event_lookup = events.set_index("事件编号")
    task_fields = {
        "风险类型": "风险类型",
        "风险等级": "风险等级",
        "风险分": "风险分",
        "影响金额": "风险金额",
        "关键证据": "关键指标",
        "置信说明": "置信说明",
        "业务类型": "业务类型",
        "动态到期阶段": "动态到期阶段",
        "客户履约等级": "客户履约等级",
        "处置强度": "处置强度",
    }
    for target, source in task_fields.items():
        tasks[target] = tasks["风险事件编号"].map(event_lookup[source]).fillna("")
    tasks["审批建议"] = tasks.apply(
        lambda row: "必须人工审批；系统不得自动停发、调额或进入法务程序"
        if row["处置强度"] in {"停发审批", "法务复核", "严格管控"}
        else "由业务人员复核后决定是否执行",
        axis=1,
    )
    tasks["建议负责人角色"] = tasks["动态到期阶段"].map(_suggested_owner_role)
    tasks["SLA状态"] = "SLA内待处理"

    added_customer_columns = [
        column for column in CUSTOMER_DERIVED_COLUMNS if column != "近180天采购总额"
    ]
    base_customer_columns = [
        column
        for column in pd.read_csv(feishu_dir / "企业渠道客户.csv", nrows=0).columns
        if column not in added_customer_columns
    ]
    customer_output = customer_numeric[
        ["客户主键", *base_customer_columns, *[c for c in added_customer_columns if c != "客户主键"]]
    ]
    evidence = pd.read_csv(feishu_dir / "企业订单证据.csv", dtype="string").fillna("")
    final_customer_ids = set(customer_output["客户编号"].astype(str))
    final_event_ids = set(events["事件编号"].astype(str))
    cross_reference_audit = {
        "customer_master_rows": int(len(customer_output)),
        "customer_master_unique_ids": int(customer_output["客户编号"].nunique()),
        "customers_added_to_master": int(len(missing_customer_ids)),
        "pre_fix_exposure_orphan_rows": int(len(original_orphan_exposure)),
        "pre_fix_exposure_orphan_customers": int(original_orphan_exposure["客户编号"].nunique()),
        "pre_fix_exposure_orphan_amount": round(
            float(original_orphan_exposure["采购风险暴露金额"].sum()), 2
        ),
        "risk_event_orphan_customer_rows": int((~events["客户编号"].isin(final_customer_ids)).sum()),
        "task_orphan_event_rows": int((~tasks["风险事件编号"].isin(final_event_ids)).sum()),
        "evidence_orphan_event_rows": int((~evidence["风险事件编号"].isin(final_event_ids)).sum()),
        "dynamic_orphan_customer_rows": int((~dynamic["客户编号"].isin(final_customer_ids)).sum()),
        "exposure_orphan_customer_rows": int((~exposure["客户编号"].isin(final_customer_ids)).sum()),
    }
    orphan_fields = [key for key in cross_reference_audit if key.endswith("_orphan_customer_rows") or key.endswith("_orphan_event_rows")]
    cross_reference_audit["status"] = (
        "pass" if all(cross_reference_audit[key] == 0 for key in orphan_fields) else "fail"
    )
    customer_output.to_csv(feishu_dir / "企业渠道客户.csv", index=False, encoding="utf-8-sig")
    events.to_csv(feishu_dir / "企业风险事件.csv", index=False, encoding="utf-8-sig")
    tasks.to_csv(feishu_dir / "企业处置任务.csv", index=False, encoding="utf-8-sig")
    dynamic.to_csv(feishu_dir / "企业动态回款监控.csv", index=False, encoding="utf-8-sig")
    exposure.to_csv(feishu_dir / "企业客户商品风险暴露.csv", index=False, encoding="utf-8-sig")

    report = {
        "status": "pass" if cross_reference_audit["status"] == "pass" else "fail",
        "rule_version": RULE_VERSION,
        "health_score_version": SCORE_VERSION,
        "snapshot_date": snapshot_date.date().isoformat(),
        "dynamic_monitor_rows": int(len(dynamic)),
        "due_soon_events_created_this_run": int(len(due_rows)),
        "due_soon_event_rows": int((events["风险类型"] == "到期前5天付款提醒").sum()),
        "risk_events_after_alignment": int(len(events)),
        "dynamic_stage_counts": {
            str(key): int(value) for key, value in dynamic["动态到期阶段"].value_counts().items()
        },
        "business_type_counts": {
            str(key): int(value) for key, value in dynamic["业务类型"].value_counts().items()
        },
        "health_customers": int(len(customer_output)),
        "exposure_rows": int(len(exposure)),
        "run_mode": run_mode,
        "run_date": run_date.date().isoformat(),
        "calculation_date": calculation_date.date().isoformat(),
        "data_age_days": data_age_days,
        "max_data_age_days": max_data_age_days,
        "stale_for_business": stale_for_business,
        "cross_reference_audit": cross_reference_audit,
        "prototype_scope": prototype_scope,
        "prototype_scope_customers": int(len(set(customer_output.loc[customer_output["试点范围命中"] == "是", "客户编号"]))),
        "boundaries": [
            "120天模型仅用于未到期开放订单风险排序，不解释为精确违约概率",
            "动态到期规则为入围赛原型参数，后续由企业校准",
            "客户商品风险为采购暴露，不代表客户实际持有库存",
            "停发、授信和法务动作只生成审批建议，不自动执行",
        ],
        "outputs": {
            "dynamic_receivables": portable_path(feishu_dir / "企业动态回款监控.csv"),
            "customer_product_exposure": portable_path(feishu_dir / "企业客户商品风险暴露.csv"),
            "customer_health": portable_path(feishu_dir / "企业渠道客户.csv"),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="补齐动态到期、五维健康和库存采购暴露原型")
    parser.add_argument("company_dir", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument("feishu_dir", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--scope-json", default="{}")
    args = parser.parse_args()
    report = build_prototype_alignment(
        args.company_dir,
        args.features,
        args.processed_dir,
        args.feishu_dir,
        args.report,
        json.loads(args.scope_json),
    )
    print(
        f"原目标对齐完成：{report['health_customers']}个客户健康诊断，"
        f"{report['dynamic_monitor_rows']}条动态到期记录，{report['exposure_rows']}条采购暴露。"
    )


if __name__ == "__main__":
    main()
