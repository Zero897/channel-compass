from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from project_paths import PROJECT_ROOT, portable_path


SOURCE_LABEL = "AFFT企业提供脱敏模拟数据_客户级聚合"
OPEN_STATUS = "insufficient_open_receivable"


def _normalize_identifier(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def _money(value: float) -> str:
    return f"¥{value:,.0f}"


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _risk_level(overdue_ratio: float, overdue_60: float, max_days: float) -> str:
    if overdue_60 > 0 or max_days > 60 or overdue_ratio >= 0.50:
        return "红色"
    return "黄色"


def _load_latest_ar(path: Path) -> tuple[str, pd.DataFrame]:
    columns = [
        "快照时间",
        "客户编号",
        "销售订单号",
        "应收金额",
        "超期应收金额",
        "超期30天以上金额",
        "超期60天以上金额",
        "超期天数",
        "最终承诺还款日期",
    ]
    ar = pd.read_csv(
        path,
        usecols=columns,
        dtype={"客户编号": "string", "销售订单号": "string", "快照时间": "string"},
        low_memory=False,
    )
    ar["快照时间"] = ar["快照时间"].str[:10]
    latest_snapshot = str(ar["快照时间"].max())
    ar = ar[ar["快照时间"] == latest_snapshot].copy()
    ar["customer_id"] = _normalize_identifier(ar["客户编号"])
    ar["order_id"] = _normalize_identifier(ar["销售订单号"])
    numeric = columns[3:8]
    for name in numeric:
        ar[name] = pd.to_numeric(ar[name], errors="coerce").fillna(0.0)
    ar["最终承诺还款日期"] = ar["最终承诺还款日期"].astype("string").str[:10]
    ar["overdue_promise_date"] = ar["最终承诺还款日期"].where(
        ar["超期应收金额"] > 0
    )
    order_ar = (
        ar.groupby(["customer_id", "order_id"], as_index=False)
        .agg(
            current_ar_amount=("应收金额", "sum"),
            overdue_ar_amount=("超期应收金额", "sum"),
            overdue_30_amount=("超期30天以上金额", "sum"),
            overdue_60_amount=("超期60天以上金额", "sum"),
            max_overdue_days=("超期天数", "max"),
            earliest_overdue_promise_date=("overdue_promise_date", "min"),
        )
    )
    return latest_snapshot, order_ar


def _customer_names(path: Path) -> dict[str, str]:
    frame = pd.read_csv(
        path,
        usecols=["客户编号_中台", "客户名称"],
        dtype={"客户编号_中台": "string", "客户名称": "string"},
    )
    frame["customer_id"] = _normalize_identifier(frame["客户编号_中台"])
    return dict(zip(frame["customer_id"], frame["客户名称"].fillna("匿名客户")))


def _build_customer_overview(
    order_ar: pd.DataFrame,
    predictive_orders: pd.DataFrame,
    names: dict[str, str],
    latest_snapshot: str,
) -> pd.DataFrame:
    ar_customer = (
        order_ar.groupby("customer_id", as_index=False)
        .agg(
            current_ar_amount=("current_ar_amount", "sum"),
            overdue_ar_amount=("overdue_ar_amount", "sum"),
            overdue_30_amount=("overdue_30_amount", "sum"),
            overdue_60_amount=("overdue_60_amount", "sum"),
            max_overdue_days=("max_overdue_days", "max"),
            earliest_overdue_promise_date=("earliest_overdue_promise_date", "min"),
            current_open_order_count=("order_id", "nunique"),
        )
    )
    predictive_customer = (
        predictive_orders.groupby("customer_id", as_index=False)
        .agg(
            predictive_high_risk_order_count=("order_id", "nunique"),
            predictive_high_risk_ar_amount=("current_ar_amount", "sum"),
            risk_weighted_ar_exposure=("risk_weighted_ar_exposure", "sum"),
            max_risk_probability=("risk_probability", "max"),
            max_risk_percentile=("risk_percentile", "max"),
        )
    )
    overview = ar_customer.merge(predictive_customer, on="customer_id", how="left")
    predictive_columns = [
        "predictive_high_risk_order_count",
        "predictive_high_risk_ar_amount",
        "risk_weighted_ar_exposure",
        "max_risk_probability",
        "max_risk_percentile",
    ]
    overview[predictive_columns] = overview[predictive_columns].fillna(0.0)
    overview["predictive_high_risk_order_count"] = overview[
        "predictive_high_risk_order_count"
    ].astype(int)
    overview["overdue_ratio"] = (
        overview["overdue_ar_amount"] / overview["current_ar_amount"].replace(0, pd.NA)
    ).fillna(0.0)
    overview["risk_level"] = overview.apply(
        lambda row: (
            "红色"
            if row["predictive_high_risk_order_count"] > 0
            or _risk_level(
                float(row["overdue_ratio"]),
                float(row["overdue_60_amount"]),
                float(row["max_overdue_days"]),
            )
            == "红色"
            else "黄色" if row["overdue_ar_amount"] > 0 else "绿色"
        ),
        axis=1,
    )
    overview["risk_source"] = overview.apply(
        lambda row: "＋".join(
            source
            for source, present in (
                ("模型预测", row["predictive_high_risk_order_count"] > 0),
                ("存量逾期", row["overdue_ar_amount"] > 0),
            )
            if present
        )
        or "未触发",
        axis=1,
    )
    overview.insert(0, "snapshot_date", latest_snapshot)
    overview.insert(2, "customer_name", overview["customer_id"].map(names).fillna("匿名客户"))
    overview["data_source"] = SOURCE_LABEL
    overview["population_scope"] = "当前有应收余额的客户"
    overview["_risk_rank"] = overview["risk_level"].map({"红色": 0, "黄色": 1, "绿色": 2})
    return overview.sort_values(
        ["_risk_rank", "overdue_ar_amount", "predictive_high_risk_ar_amount"],
        ascending=[True, False, False],
    ).drop(columns="_risk_rank")


def _events_and_tasks(
    overview: pd.DataFrame,
    predictive_orders: pd.DataFrame,
    latest_snapshot: str,
    model_name: str,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    generated_date = date.today()
    due_date = generated_date + timedelta(days=2)
    events: list[dict[str, object]] = []
    for row in overview.to_dict("records"):
        customer_id = str(row["customer_id"])
        history = (
            f"当前应收{_money(float(row['current_ar_amount']))}，"
            f"逾期占比{_percent(float(row['overdue_ratio']))}，"
            f"模型高风险开放订单{int(row['predictive_high_risk_order_count'])}笔"
        )
        if int(row["predictive_high_risk_order_count"]) > 0:
            customer_orders = predictive_orders[
                predictive_orders["customer_id"] == customer_id
            ].sort_values("risk_probability", ascending=False)
            top_reasons = customer_orders["evidence"].drop_duplicates().head(1).tolist()
            evidence = (
                f"最高风险分位{float(row['max_risk_percentile']):.1f}/100；"
                f"高风险开放应收{_money(float(row['predictive_high_risk_ar_amount']))}；"
                + "；".join(top_reasons)
            )
            risk_type = (
                "存量风险客户新增订单风险扩散预警"
                if float(row["overdue_ar_amount"]) > 0
                else "开放订单风险排序预警"
            )
            events.append(
                {
                    "事件编号": f"AFFT-{latest_snapshot.replace('-', '')}-{customer_id}-PRED",
                    "客户编号": customer_id,
                    "客户名称": row["customer_name"],
                    "风险类型": risk_type,
                    "风险等级": "红色",
                    "触发时间": generated_date.isoformat(),
                    "数据快照": latest_snapshot,
                    "关键指标": evidence,
                    "风险金额": round(float(row["predictive_high_risk_ar_amount"]), 2),
                    "风险加权应收暴露": round(float(row["risk_weighted_ar_exposure"]), 2),
                    "模型概率": round(float(row["max_risk_probability"]), 6),
                    "风险分": round(float(row["max_risk_percentile"]), 2),
                    "置信说明": (
                        f"{model_name}输出超过冻结阈值{threshold:.6f}；"
                        "风险分表示当前开放订单中的相对分位，模型概率不解释为精确违约率"
                    ),
                    "客户历史摘要": history,
                    "规则解释": (
                        "订单本身当前未逾期；客户可能已有其他存量逾期。"
                        "本事件用于识别风险向新增订单扩散，不等同于客户首次逾期前预警"
                    ),
                    "AI风险摘要": "",
                    "建议动作": "核查高风险订单的付款条件和回款计划，联系业务负责人确认客户近期资金安排",
                    "处理状态": "待处理",
                    "模型版本": model_name,
                    "模型阈值": round(threshold, 6),
                    "data_source": SOURCE_LABEL,
                }
            )
        if float(row["overdue_ar_amount"]) > 0:
            level = _risk_level(
                float(row["overdue_ratio"]),
                float(row["overdue_60_amount"]),
                float(row["max_overdue_days"]),
            )
            evidence = (
                f"逾期应收{_money(float(row['overdue_ar_amount']))}，"
                f"占当前应收{_percent(float(row['overdue_ratio']))}；"
                f"60天以上{_money(float(row['overdue_60_amount']))}；"
                f"最长逾期{float(row['max_overdue_days']):.0f}天；"
                f"最早到期日{row['earliest_overdue_promise_date']}"
            )
            events.append(
                {
                    "事件编号": f"AFFT-{latest_snapshot.replace('-', '')}-{customer_id}-OVERDUE",
                    "客户编号": customer_id,
                    "客户名称": row["customer_name"],
                    "风险类型": "存量逾期应收",
                    "风险等级": level,
                    "触发时间": generated_date.isoformat(),
                    "数据快照": latest_snapshot,
                    "关键指标": evidence,
                    "风险金额": round(float(row["overdue_ar_amount"]), 2),
                    "风险加权应收暴露": "",
                    "模型概率": "",
                    "风险分": "",
                    "置信说明": "应收快照确定性规则命中，不是模型预测",
                    "客户历史摘要": history,
                    "规则解释": f"快照实际读取的最早最终承诺还款日为{row['earliest_overdue_promise_date']}，且存在超期应收金额",
                    "AI风险摘要": "",
                    "建议动作": "核对应收明细、最终承诺还款日及展期记录，确认催收责任人和回款节点",
                    "处理状态": "待处理",
                    "模型版本": "business_rule",
                    "模型阈值": "",
                    "data_source": SOURCE_LABEL,
                }
            )
    event_frame = pd.DataFrame(events).sort_values(
        ["风险等级", "风险金额"], ascending=[True, False]
    )
    tasks: list[dict[str, object]] = []
    for event in event_frame.to_dict("records"):
        if event["风险等级"] != "红色":
            continue
        digest = hashlib.sha256(str(event["事件编号"]).encode("utf-8")).hexdigest()[:12].upper()
        tasks.append(
            {
                "任务编号": f"TASK-{digest}",
                "风险事件编号": event["事件编号"],
                "客户编号": event["客户编号"],
                "客户名称": event["客户名称"],
                "AI建议动作": event["建议动作"],
                "人工处置方案": "待填写",
                "负责人": "",
                "创建时间": generated_date.isoformat(),
                "截止日期": due_date.isoformat(),
                "审批状态": "待审批",
                "执行状态": "待处理",
                "完成时间": "",
                "实际回款金额": "",
                "执行结果": "",
                "预警有效性": "待确认",
                "反馈备注": "",
                "data_source": SOURCE_LABEL,
            }
        )
    return event_frame, pd.DataFrame(tasks)


def _preserve_task_state(tasks: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    if not existing_path.exists() or tasks.empty:
        return tasks
    existing = pd.read_csv(existing_path, dtype="string").fillna("")
    if "任务编号" not in existing.columns:
        return tasks
    state_fields = [
        "创建时间",
        "截止日期",
        "人工处置方案",
        "负责人",
        "审批状态",
        "执行状态",
        "完成时间",
        "实际回款金额",
        "执行结果",
        "预警有效性",
        "反馈备注",
    ]
    available = [field for field in state_fields if field in existing.columns]
    if not available:
        return tasks
    prior = existing[["任务编号", *available]].drop_duplicates("任务编号", keep="last")
    merged = tasks.merge(prior, on="任务编号", how="left", suffixes=("", "_历史"))
    for field in available:
        historical = f"{field}_历史"
        historical_value = merged[historical].fillna("").astype(str).str.strip()
        mask = historical_value != ""
        if field == "负责人":
            mask &= historical_value != "待分配"
        merged.loc[mask, field] = merged.loc[mask, historical]
    return merged.drop(columns=[f"{field}_历史" for field in available])


def _preserve_event_state(events: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    if not existing_path.exists() or events.empty:
        return events
    existing = pd.read_csv(existing_path, dtype="string").fillna("")
    if "事件编号" not in existing.columns or "触发时间" not in existing.columns:
        return events
    prior = existing[["事件编号", "触发时间"]].drop_duplicates("事件编号", keep="last")
    merged = events.merge(prior, on="事件编号", how="left", suffixes=("", "_历史"))
    historical = merged["触发时间_历史"].fillna("").astype(str).str.strip()
    mask = historical != ""
    merged.loc[mask, "触发时间"] = merged.loc[mask, "触发时间_历史"]
    return merged.drop(columns=["触发时间_历史"])


def _write_demo_task_cases(tasks: pd.DataFrame, path: Path) -> None:
    demo = tasks.head(3).copy()
    if len(demo) < 3:
        return
    demo["记录性质"] = "演示处置记录（不计入真实闭环率）"
    demo.loc[demo.index[0], ["审批状态", "执行状态", "人工处置方案", "执行结果", "预警有效性"]] = [
        "已批准",
        "已完成",
        "核对账期并确认分期回款节点",
        "演示：已确认下一回款日期",
        "有效",
    ]
    demo.loc[demo.index[1], ["审批状态", "执行状态", "人工处置方案", "执行结果", "预警有效性"]] = [
        "已驳回",
        "已关闭",
        "演示：补充客户近期资金证明后再审批",
        "演示：方案退回调整",
        "待确认",
    ]
    demo.loc[demo.index[2], ["审批状态", "执行状态", "人工处置方案", "执行结果", "预警有效性"]] = [
        "已批准",
        "处理中",
        "演示：业务与财务联合核查开放订单",
        "演示：正在联系客户",
        "待确认",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(path, index=False, encoding="utf-8-sig")


def _write_judge_cases(
    path: Path,
    overview: pd.DataFrame,
    predictive_orders: pd.DataFrame,
) -> None:
    predictive_candidates = overview[overview["predictive_high_risk_order_count"] > 0]
    overdue_candidates = overview[overview["overdue_ar_amount"] > 0]
    green_candidates = overview[overview["risk_level"] == "绿色"]
    lines = [
        "# 评委演示案例",
        "",
        "> 案例均来自企业提供的脱敏模拟数据。当前未找到“客户历史从未逾期、评分后首次发生逾期”的严格客户级提前预警样例，因此不作该项宣称。",
        "",
    ]
    if not predictive_candidates.empty:
        candidate = predictive_candidates.sort_values(
            ["overdue_ar_amount", "max_risk_probability"], ascending=[True, False]
        ).iloc[0]
        orders = predictive_orders[
            predictive_orders["customer_id"] == candidate["customer_id"]
        ].sort_values("risk_probability", ascending=False)
        evidence = str(orders.iloc[0]["evidence"]) if not orders.empty else "需人工复核"
        lines.extend(
            [
                "## 案例一：存量风险向新增订单扩散监测",
                "",
                f"- 客户：{candidate['customer_id']}｜{candidate['customer_name']}。",
                f"- 订单自身尚未逾期但进入高风险队列：{int(candidate['predictive_high_risk_order_count'])}笔，最高风险分位{float(candidate['max_risk_percentile']):.1f}/100。",
                f"- 客户存量状态：逾期应收{_money(float(candidate['overdue_ar_amount']))}；因此这是风险扩散监测，不是客户首次逾期前预警。",
                f"- 真实模型贡献证据：{evidence}。",
                "- 演示动作：查看订单证据，生成人工核查任务，不直接调整授信。",
                "",
            ]
        )
    if not overdue_candidates.empty:
        candidate = overdue_candidates.sort_values("overdue_ar_amount", ascending=False).iloc[0]
        lines.extend(
            [
                "## 案例二：存量逾期处置",
                "",
                f"- 客户：{candidate['customer_id']}｜{candidate['customer_name']}。",
                f"- 逾期应收：{_money(float(candidate['overdue_ar_amount']))}，最长逾期{float(candidate['max_overdue_days']):.0f}天。",
                f"- 快照证据：最早最终承诺还款日{candidate['earliest_overdue_promise_date']}。",
                "- 演示动作：核对应收明细和展期记录，进入审批与催收任务。",
                "",
            ]
        )
    if not green_candidates.empty:
        candidate = green_candidates.sort_values("current_ar_amount").iloc[0]
        lines.extend(
            [
                "## 案例三：绿色对照客户",
                "",
                f"- 客户：{candidate['customer_id']}｜{candidate['customer_name']}。",
                f"- 当前应收：{_money(float(candidate['current_ar_amount']))}；未触发开放订单预测或存量逾期规则。",
                "- 演示价值：证明系统保留正常客户，不把所有客户都判为高风险。",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_customer_risk(
    company_dir: Path,
    features_path: Path,
    scores_path: Path,
    model_metrics_path: Path,
    processed_dir: Path,
    feishu_dir: Path,
    audit_path: Path,
) -> dict[str, object]:
    model_report = json.loads(model_metrics_path.read_text(encoding="utf-8"))
    model_name = str(model_report["primary_model"])
    threshold = float(model_report["primary_threshold"])
    if not features_path.exists():
        raise FileNotFoundError(f"缺少订单特征文件：{features_path}")
    scores = pd.read_csv(
        scores_path,
        dtype={"order_id": "string", "customer_id": "string", "label_status": "string"},
    )
    scores["order_id"] = _normalize_identifier(scores["order_id"])
    scores["customer_id"] = _normalize_identifier(scores["customer_id"])
    open_scores = scores[scores["label_status"] == OPEN_STATUS].copy()
    latest_snapshot, order_ar = _load_latest_ar(company_dir / "应收快照_月末24期.csv")
    joined = open_scores.merge(order_ar, on=["customer_id", "order_id"], how="left", indicator=True)
    matched = int((joined["_merge"] == "both").sum())
    coverage = matched / len(joined) if len(joined) else 0.0
    if coverage < 0.99:
        raise ValueError(f"开放订单到最新应收关联覆盖率过低：{coverage:.2%}")
    joined = joined[joined["_merge"] == "both"].drop(columns="_merge")
    joined["risk_percentile"] = joined["risk_probability"].rank(
        pct=True, method="average"
    ) * 100.0
    predictive_orders = joined[joined["high_risk"] == 1].copy()
    predictive_orders["risk_weighted_ar_exposure"] = (
        predictive_orders["risk_probability"] * predictive_orders["current_ar_amount"]
    )
    predictive_orders["evidence"] = predictive_orders["model_top_contributions"].fillna(
        "模型未输出有效贡献，需人工复核"
    )
    names = _customer_names(company_dir / "客户授信.csv")
    overview = _build_customer_overview(order_ar, predictive_orders, names, latest_snapshot)
    events, tasks = _events_and_tasks(
        overview, predictive_orders, latest_snapshot, model_name, threshold
    )
    events = _preserve_event_state(events, feishu_dir / "企业风险事件.csv")
    tasks = _preserve_task_state(tasks, feishu_dir / "企业处置任务.csv")
    _write_demo_task_cases(tasks, PROJECT_ROOT / "demo_output" / "企业处置任务_演示记录.csv")
    _write_judge_cases(
        processed_dir.parents[1] / "docs" / "judge_cases.md",
        overview,
        predictive_orders,
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    feishu_dir.mkdir(parents=True, exist_ok=True)
    overview.to_csv(
        processed_dir / "company_customer_risk_overview.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictive_orders.sort_values(
        ["customer_id", "risk_probability"], ascending=[True, False]
    ).to_csv(
        processed_dir / "company_risk_order_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    customer_export = overview.rename(
        columns={
            "snapshot_date": "快照日期",
            "customer_id": "客户编号",
            "customer_name": "客户名称",
            "current_ar_amount": "当前应收余额",
            "overdue_ar_amount": "逾期应收金额",
            "overdue_30_amount": "逾期30天以上金额",
            "overdue_60_amount": "逾期60天以上金额",
            "max_overdue_days": "最大逾期天数",
            "earliest_overdue_promise_date": "最早最终承诺还款日",
            "current_open_order_count": "当前开放订单数",
            "overdue_ratio": "逾期占比",
            "predictive_high_risk_order_count": "预测高风险订单数",
            "predictive_high_risk_ar_amount": "预测高风险应收金额",
            "risk_weighted_ar_exposure": "风险加权应收暴露",
            "max_risk_probability": "最高模型概率",
            "max_risk_percentile": "最高风险分",
            "risk_level": "风险等级",
            "risk_source": "风险来源",
            "data_source": "数据来源",
            "population_scope": "客户统计口径",
        }
    )
    customer_export.insert(
        0,
        "客户主键",
        customer_export["客户编号"].astype(str) + "｜" + customer_export["客户名称"].astype(str),
    )
    customer_export.to_csv(
        feishu_dir / "企业渠道客户.csv", index=False, encoding="utf-8-sig"
    )
    events.to_csv(feishu_dir / "企业风险事件.csv", index=False, encoding="utf-8-sig")
    tasks.to_csv(feishu_dir / "企业处置任务.csv", index=False, encoding="utf-8-sig")
    top_evidence = (
        predictive_orders.sort_values(
            ["customer_id", "risk_probability"], ascending=[True, False]
        )
        .groupby("customer_id", as_index=False)
        .head(3)
        .rename(
            columns={
                "customer_id": "客户编号",
                "order_id": "销售订单号",
                "order_date": "出库日期",
                "risk_probability": "风险概率",
                "risk_percentile": "风险分",
                "current_ar_amount": "当前应收金额",
                "risk_weighted_ar_exposure": "风险加权应收暴露",
                "evidence": "关键证据",
            }
        )
    )
    top_evidence["风险事件编号"] = top_evidence["客户编号"].map(
        lambda customer_id: f"AFFT-{latest_snapshot.replace('-', '')}-{customer_id}-PRED"
    )
    top_evidence["订单证据编号"] = top_evidence.apply(
        lambda row: "EVD-"
        + hashlib.sha256(
            f"{row['风险事件编号']}|{row['销售订单号']}".encode("utf-8")
        ).hexdigest()[:12].upper(),
        axis=1,
    )
    top_evidence[
        ["订单证据编号", "风险事件编号", "客户编号", "销售订单号", "出库日期", "风险分", "风险概率", "当前应收金额", "风险加权应收暴露", "关键证据"]
    ].to_csv(feishu_dir / "企业订单证据.csv", index=False, encoding="utf-8-sig")

    metric_rows: list[dict[str, object]] = []
    for result in model_report["models"]:
        metrics = result["splits"]["test"]
        for metric in (
            "pr_auc",
            "roc_auc",
            "precision",
            "recall",
            "false_positive_rate",
            "business_false_alarm_rate",
            "risk_amount_capture",
            "top20_risk_amount_capture",
        ):
            metric_row = {
                    "模型名称": result["model"],
                    "时间区间": "固定120天观察期的时间外测试集",
                    "指标名称": metric,
                    "指标值": metrics[metric],
                    "是否主模型": "是" if result["model"] == model_name else "否",
                    "数据来源": SOURCE_LABEL,
                }
            metric_row["指标编号"] = "MET-" + hashlib.sha256(
                f"{metric_row['模型名称']}|{metric_row['时间区间']}|{metric}".encode("utf-8")
            ).hexdigest()[:12].upper()
            metric_rows.append(metric_row)
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame = metric_frame[["指标编号", *[column for column in metric_frame.columns if column != "指标编号"]]]
    metric_frame.to_csv(
        feishu_dir / "企业模型指标.csv", index=False, encoding="utf-8-sig"
    )

    audit = {
        "status": "pass",
        "latest_ar_snapshot": latest_snapshot,
        "primary_model": model_name,
        "frozen_threshold": threshold,
        "holdout_scored_orders": len(scores),
        "open_not_overdue_orders": len(open_scores),
        "open_orders_joined_to_latest_ar": matched,
        "join_coverage": round(coverage, 6),
        "predictive_high_risk_orders": len(predictive_orders),
        "predictive_high_risk_customers": int(predictive_orders["customer_id"].nunique()),
        "latest_ar_customers": int(overview["customer_id"].nunique()),
        "customers_with_existing_overdue": int((overview["overdue_ar_amount"] > 0).sum()),
        "risk_events": len(events),
        "red_events": int((events["风险等级"] == "红色").sum()),
        "tasks": len(tasks),
        "event_boundary": {
            "prediction": "仅订单自身尚未逾期且模型输出超过冻结阈值；客户可能已有其他存量逾期，当前定位为风险扩散监测",
            "existing_overdue": "最新应收快照中已经存在超期应收金额",
        },
        "data_safety": "只向飞书导出客户级聚合和最多每客户3笔订单证据，不导出原始流水",
        "risk_weighted_exposure_policy": "校准后的订单超期概率乘当前应收，仅表示风险加权应收暴露，不代表预期坏账损失",
        "display_policy": "界面优先展示0-100风险分位；模型概率仅保留作审计和风险加权测算，不解释为精确违约率",
        "customer_population": "当前有应收余额的客户",
        "outputs": {
            "customer_overview": portable_path(feishu_dir / "企业渠道客户.csv"),
            "risk_events": portable_path(feishu_dir / "企业风险事件.csv"),
            "tasks": portable_path(feishu_dir / "企业处置任务.csv"),
            "order_evidence": portable_path(feishu_dir / "企业订单证据.csv"),
            "model_metrics": portable_path(feishu_dir / "企业模型指标.csv"),
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="将订单模型概率聚合为客户风险与飞书导入表")
    parser.add_argument("company_dir", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("scores", type=Path)
    parser.add_argument("model_metrics", type=Path)
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument("feishu_dir", type=Path)
    parser.add_argument("audit", type=Path)
    args = parser.parse_args()
    audit = aggregate_customer_risk(
        args.company_dir,
        args.features,
        args.scores,
        args.model_metrics,
        args.processed_dir,
        args.feishu_dir,
        args.audit,
    )
    print(
        f"客户风险聚合完成：{audit['latest_ar_customers']}家客户，"
        f"{audit['risk_events']}条风险事件，{audit['tasks']}条红色处置任务。"
    )


if __name__ == "__main__":
    main()
