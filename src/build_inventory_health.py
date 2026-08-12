from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import portable_path


SOURCE_LABEL = "AFFT企业提供脱敏模拟数据_SKU仓库聚合"


def _normalize_identifier(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def _risk_level(row: pd.Series) -> str:
    has_recent_sales = row["sales_quantity_recent"] > 0
    if (
        row["stale_365_ratio"] >= 0.20
        or row["overdue_borrow_amount"] > 0
        or (has_recent_sales and row["coverage_days"] > 180)
        or (not has_recent_sales and row["max_inventory_age"] >= 365)
    ):
        return "红色"
    if (
        row["stale_180_ratio"] >= 0.10
        or (has_recent_sales and row["coverage_days"] > 90)
        or (not has_recent_sales and row["max_inventory_age"] >= 180)
        or row["sales_growth_90d"] <= -0.30
    ):
        return "黄色"
    return "绿色"


def _evidence(row: pd.Series) -> str:
    coverage = (
        f"库存覆盖{row['coverage_days']:.0f}天"
        if row["sales_quantity_recent"] > 0
        else "近90天无匹配销量"
    )
    return (
        f"库存金额¥{row['inventory_value']:,.0f}；"
        f"180天以上占比{row['stale_180_ratio']:.1%}；"
        f"365天以上占比{row['stale_365_ratio']:.1%}；"
        f"{coverage}；"
        f"近90天销售变化{row['sales_growth_90d']:.1%}"
    )


def build_inventory_health(
    company_dir: Path,
    processed_dir: Path,
    feishu_dir: Path,
    report_path: Path,
) -> dict[str, object]:
    inventory_path = company_dir / "库龄快照_季末8期.csv"
    sales_path = company_dir / "销售流水.csv"
    inventory_columns = [
        "快照日期",
        "物料编码",
        "物料大类名称",
        "物料小类名称",
        "产品线名称",
        "库存组织名称",
        "数量",
        "含税总价",
        "库龄",
        "是否超期",
    ]
    inventory = pd.read_csv(
        inventory_path,
        usecols=inventory_columns,
        dtype={"快照日期": "string", "物料编码": "string", "库存组织名称": "string"},
        low_memory=False,
    )
    inventory["快照日期"] = inventory["快照日期"].str[:10]
    latest_snapshot = str(inventory["快照日期"].max())
    inventory = inventory[inventory["快照日期"] == latest_snapshot].copy()
    inventory["sku_id"] = _normalize_identifier(inventory["物料编码"])
    inventory["warehouse"] = inventory["库存组织名称"].fillna("未知仓库")
    for column in ("数量", "含税总价", "库龄"):
        inventory[column] = pd.to_numeric(inventory[column], errors="coerce").fillna(0.0)
    inventory["inventory_value"] = inventory["含税总价"].clip(lower=0)
    inventory["stale_180_amount"] = inventory["inventory_value"].where(
        inventory["库龄"] >= 180, 0.0
    )
    inventory["stale_365_amount"] = inventory["inventory_value"].where(
        inventory["库龄"] >= 365, 0.0
    )
    inventory["overdue_borrow_amount"] = inventory["inventory_value"].where(
        inventory["是否超期"].astype("string").str.upper().isin({"Y", "YES", "是", "1", "TRUE"}),
        0.0,
    )
    current = (
        inventory.groupby(["sku_id", "warehouse"], as_index=False)
        .agg(
            product_line=("产品线名称", "first"),
            product_category=("物料大类名称", "first"),
            inventory_quantity=("数量", "sum"),
            inventory_value=("inventory_value", "sum"),
            stale_180_amount=("stale_180_amount", "sum"),
            stale_365_amount=("stale_365_amount", "sum"),
            overdue_borrow_amount=("overdue_borrow_amount", "sum"),
            max_inventory_age=("库龄", "max"),
        )
    )

    sales_columns = ["出库日期", "物料编码", "库存组织名称", "数量", "销售金额_折扣后_含税"]
    sales = pd.read_csv(
        sales_path,
        usecols=sales_columns,
        dtype={"出库日期": "string", "物料编码": "string", "库存组织名称": "string"},
        low_memory=False,
    )
    sales["sale_date"] = pd.to_datetime(sales["出库日期"].str[:10], errors="coerce")
    snapshot_date = pd.Timestamp(latest_snapshot)
    window_180 = pd.to_timedelta(180, unit="D")
    window_90 = pd.to_timedelta(90, unit="D")
    sales = sales[
        (sales["sale_date"] > snapshot_date - window_180)
        & (sales["sale_date"] <= snapshot_date)
    ].copy()
    sales["sku_id"] = _normalize_identifier(sales["物料编码"])
    sales["warehouse"] = sales["库存组织名称"].fillna("未知仓库")
    sales["数量"] = pd.to_numeric(sales["数量"], errors="coerce").fillna(0.0).clip(lower=0)
    sales["销售金额_折扣后_含税"] = pd.to_numeric(
        sales["销售金额_折扣后_含税"], errors="coerce"
    ).fillna(0.0).clip(lower=0)
    sales["period"] = np.where(
        sales["sale_date"] > snapshot_date - window_90, "recent", "previous"
    )
    trend = (
        sales.groupby(["sku_id", "warehouse", "period"], as_index=False)
        .agg(sales_quantity=("数量", "sum"), sales_amount=("销售金额_折扣后_含税", "sum"))
        .pivot(index=["sku_id", "warehouse"], columns="period", values=["sales_quantity", "sales_amount"])
        .fillna(0.0)
    )
    trend.columns = [f"{metric}_{period}" for metric, period in trend.columns]
    trend = trend.reset_index()
    for column in (
        "sales_quantity_recent",
        "sales_quantity_previous",
        "sales_amount_recent",
        "sales_amount_previous",
    ):
        if column not in trend:
            trend[column] = 0.0

    result = current.merge(trend, on=["sku_id", "warehouse"], how="left").fillna(0.0)
    result["stale_180_ratio"] = (
        result["stale_180_amount"] / result["inventory_value"].replace(0, np.nan)
    ).fillna(0.0)
    result["stale_365_ratio"] = (
        result["stale_365_amount"] / result["inventory_value"].replace(0, np.nan)
    ).fillna(0.0)
    daily_sales = result["sales_quantity_recent"] / 90.0
    result["baseline_daily_demand"] = daily_sales
    result["baseline_demand_4w"] = daily_sales * 28.0
    result["baseline_demand_8w"] = daily_sales * 56.0
    result["projected_inventory_after_8w"] = (
        result["inventory_quantity"].clip(lower=0) - result["baseline_demand_8w"]
    ).clip(lower=0)
    result["baseline_shortage_8w"] = (
        result["baseline_demand_8w"] - result["inventory_quantity"].clip(lower=0)
    ).clip(lower=0)
    result["demand_baseline_method"] = "最近90天日均销量×预测天数；入围赛基线，非机器学习需求预测"
    result["coverage_days"] = np.where(
        daily_sales > 0, result["inventory_quantity"].clip(lower=0) / daily_sales, 999.0
    )
    result["sales_growth_90d"] = (
        (result["sales_amount_recent"] - result["sales_amount_previous"])
        / result["sales_amount_previous"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["risk_level"] = result.apply(_risk_level, axis=1)
    result["evidence"] = result.apply(_evidence, axis=1)
    result.insert(0, "snapshot_date", latest_snapshot)
    result["data_source"] = SOURCE_LABEL
    result["entity_scope"] = "SKU-仓库，不归因到客户"

    processed_dir.mkdir(parents=True, exist_ok=True)
    feishu_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        processed_dir / "company_inventory_health.csv", index=False, encoding="utf-8-sig"
    )
    risk_rank = result["risk_level"].map({"红色": 0, "黄色": 1, "绿色": 2})
    export = result.assign(_risk_rank=risk_rank)
    export = export[export["risk_level"] != "绿色"].sort_values(
        ["_risk_rank", "inventory_value"], ascending=[True, False]
    ).head(500)
    export = export.rename(
        columns={
            "snapshot_date": "快照日期",
            "sku_id": "物料编码",
            "warehouse": "库存组织",
            "product_line": "产品线",
            "product_category": "物料大类",
            "inventory_quantity": "库存数量",
            "inventory_value": "库存金额",
            "stale_180_amount": "180天以上库存金额",
            "stale_365_amount": "365天以上库存金额",
            "stale_180_ratio": "180天以上占比",
            "stale_365_ratio": "365天以上占比",
            "overdue_borrow_amount": "借物超期金额",
            "coverage_days": "库存覆盖天数",
            "sales_growth_90d": "近90天销售变化",
            "baseline_daily_demand": "日均需求基线",
            "baseline_demand_4w": "未来4周需求基线",
            "baseline_demand_8w": "未来8周需求基线",
            "projected_inventory_after_8w": "8周后预计剩余库存",
            "baseline_shortage_8w": "8周需求缺口基线",
            "demand_baseline_method": "需求基线方法",
            "risk_level": "风险等级",
            "evidence": "关键证据",
            "entity_scope": "对象口径",
            "data_source": "数据来源",
        }
    ).drop(columns="_risk_rank")
    export.insert(
        0,
        "库存对象编号",
        export.apply(
            lambda row: "INV-"
            + hashlib.sha256(
                f"{row['快照日期']}|{row['物料编码']}|{row['库存组织']}".encode("utf-8")
            ).hexdigest()[:12].upper(),
            axis=1,
        ),
    )
    export.to_csv(feishu_dir / "企业库存风险.csv", index=False, encoding="utf-8-sig")
    summary = (
        result.groupby(["product_line", "warehouse", "risk_level"], as_index=False)
        .agg(
            sku_warehouse_count=("sku_id", "size"),
            inventory_value=("inventory_value", "sum"),
            stale_180_amount=("stale_180_amount", "sum"),
            stale_365_amount=("stale_365_amount", "sum"),
        )
        .rename(
            columns={
                "product_line": "产品线",
                "warehouse": "库存组织",
                "risk_level": "风险等级",
                "sku_warehouse_count": "SKU仓位数",
                "inventory_value": "库存金额",
                "stale_180_amount": "180天以上库存金额",
                "stale_365_amount": "365天以上库存金额",
            }
        )
    )
    summary["统计口径"] = "全量SKU-仓库组合，明细表仅展示风险金额Top500"
    summary["数据来源"] = SOURCE_LABEL
    summary.insert(
        0,
        "库存汇总编号",
        summary.apply(
            lambda row: "INVSUM-"
            + hashlib.sha256(
                f"{latest_snapshot}|{row['产品线']}|{row['库存组织']}|{row['风险等级']}".encode("utf-8")
            ).hexdigest()[:12].upper(),
            axis=1,
        ),
    )
    summary.to_csv(
        feishu_dir / "企业库存风险汇总.csv", index=False, encoding="utf-8-sig"
    )

    audit = {
        "status": "pass",
        "latest_snapshot": latest_snapshot,
        "sku_warehouse_rows": int(len(result)),
        "red_rows": int((result["risk_level"] == "红色").sum()),
        "yellow_rows": int((result["risk_level"] == "黄色").sum()),
        "total_risk_rows": int((result["risk_level"] != "绿色").sum()),
        "feishu_export_rows": int(len(export)),
        "detail_export_limit": 500,
        "detail_export_truncated": bool((result["risk_level"] != "绿色").sum() > len(export)),
        "demand_baseline": {
            "horizons_weeks": [4, 8],
            "method": "最近90天日均销量外推",
            "backtest_proxy_wape": round(
                float(
                    (result["sales_quantity_recent"] - result["sales_quantity_previous"])
                    .abs()
                    .sum()
                    / result["sales_quantity_recent"].abs().sum()
                )
                if result["sales_quantity_recent"].abs().sum()
                else 0.0,
                6,
            ),
            "boundary": "用于体现4—8周需求预测思路的朴素基线，不宣称已完成客户—SKU机器学习预测",
        },
        "scope": "SKU-仓库库存健康，不关联或归因到渠道客户",
        "output": portable_path(feishu_dir / "企业库存风险.csv"),
        "summary_output": portable_path(feishu_dir / "企业库存风险汇总.csv"),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="构建SKU-仓库库存健康与呆滞风险表")
    parser.add_argument("company_dir", type=Path)
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument("feishu_dir", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    audit = build_inventory_health(
        args.company_dir, args.processed_dir, args.feishu_dir, args.report
    )
    print(
        f"库存健康构建完成：{audit['sku_warehouse_rows']}个SKU-仓库组合，"
        f"飞书导出{audit['feishu_export_rows']}条风险记录。"
    )


if __name__ == "__main__":
    main()
