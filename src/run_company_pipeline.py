from __future__ import annotations

import argparse
import json
from pathlib import Path

from aggregate_company_customer_risk import aggregate_customer_risk
from analyze_early_warning_evidence import analyze_early_warning
from audit_company_data import audit
from build_company_training_data import build_training_data
from build_inventory_health import build_inventory_health
from build_prototype_alignment import build_prototype_alignment
from build_process_event_log import build_process_event_log
from import_task_feedback import import_feedback
from project_paths import portable_path
from simulate_treatment_scenarios import simulate_scenarios
from train_payment_risk_models import train_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def run_company_pipeline(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    company_dir = _resolve(config["company_data_dir"])
    processed_dir = _resolve(config["processed_dir"])
    reports_dir = _resolve(config["reports_dir"])
    models_dir = _resolve(config["models_dir"])
    docs_dir = _resolve(config["docs_dir"])
    feishu_dir = _resolve(config["feishu_dir"])
    feedback_path = _resolve(config["feedback_path"]) if config.get("feedback_path") else None
    required = [
        "销售流水.csv",
        "业务回款明细.csv",
        "应收快照_月末24期.csv",
        "库龄快照_季末8期.csv",
        "增值合同签约明细.csv",
        "展期记录.csv",
        "客户授信.csv",
    ]
    missing = [name for name in required if not (company_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"企业数据目录缺少文件：{', '.join(missing)}")

    print("[1/8] 审计企业数据")
    audit_report = audit(
        company_dir,
        reports_dir / "company_data_audit.json",
        docs_dir / "company_data_audit.md",
    )
    print("[2/8] 构建120天成熟标签与无泄漏特征")
    training_data_report = build_training_data(
        company_dir,
        processed_dir,
        reports_dir / "training_data_audit.json",
    )
    print("[3/8] 训练、校准并按验证集冻结模型")
    model_report = train_models(
        processed_dir / "order_features.csv", reports_dir, models_dir, docs_dir
    )
    early_warning_report = analyze_early_warning(
        processed_dir / "order_features.csv",
        processed_dir / "order_labels.csv",
        reports_dir / "test_predictions.csv",
        reports_dir / "strict_early_warning_cases.csv",
        reports_dir / "early_warning_evidence_audit.json",
    )
    print("[4/8] 聚合客户风险并生成飞书表")
    customer_report = aggregate_customer_risk(
        company_dir,
        processed_dir / "order_features.csv",
        reports_dir / "holdout_risk_scores.csv",
        reports_dir / "model_metrics.json",
        processed_dir,
        feishu_dir,
        reports_dir / "customer_risk_aggregation_audit.json",
    )
    feedback_report: dict[str, object] = {"feedback_rows": 0}
    if feedback_path and feedback_path.exists():
        feedback_report = import_feedback(
            feishu_dir / "企业处置任务.csv",
            feedback_path,
            feishu_dir / "企业处置任务.csv",
            reports_dir / "task_feedback_audit.json",
        )
    print("[5/8] 构建SKU-仓库库存健康与4—8周需求基线")
    inventory_report = build_inventory_health(
        company_dir,
        processed_dir,
        feishu_dir,
        reports_dir / "inventory_health_audit.json",
    )
    print("[6/8] 补齐动态到期、五维健康和客户商品风险暴露")
    alignment_report = build_prototype_alignment(
        company_dir,
        processed_dir / "order_features.csv",
        processed_dir,
        feishu_dir,
        reports_dir / "prototype_alignment_audit.json",
        config.get("prototype_scope", {}),
    )
    print("[7/8] 生成三种透明处置情景")
    scenario_report = simulate_scenarios(
        feishu_dir / "企业渠道客户.csv",
        feishu_dir / "企业处置情景.csv",
        reports_dir / "scenario_simulation_audit.json",
    )
    print("[8/8] 生成风险处置事件时间线和SLA状态")
    process_report = build_process_event_log(
        feishu_dir / "企业风险事件.csv",
        feishu_dir / "企业处置任务.csv",
        feishu_dir / "企业订单证据.csv",
        feishu_dir / "企业风险处置时间线.csv",
        reports_dir / "process_event_log_audit.json",
    )
    summary = {
        "status": "pass",
        "config": portable_path(config_path),
        "data_audit": audit_report.get("status", "pass"),
        "training_data": training_data_report["status"],
        "primary_model": model_report["primary_model"],
        "strict_early_warning_cases": early_warning_report["strict_customer_early_warning_cases"],
        "customer_events": alignment_report["risk_events_after_alignment"],
        "inventory_risks_exported": inventory_report["feishu_export_rows"],
        "dynamic_receivable_rows": alignment_report["dynamic_monitor_rows"],
        "health_customers": alignment_report["health_customers"],
        "customer_product_exposure_rows": alignment_report["exposure_rows"],
        "scenario_rows": scenario_report["scenario_rows"],
        "process_events": process_report["event_rows"],
        "feedback_rows": int(feedback_report.get("feedback_rows", 0)),
    }
    (reports_dir / "company_pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="一键运行渠智罗盘企业主链")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "company_pipeline.json",
    )
    args = parser.parse_args()
    summary = run_company_pipeline(args.config)
    print(
        f"企业主链完成：主模型{summary['primary_model']}，"
        f"{summary['customer_events']}条客户风险事件，"
        f"{summary['inventory_risks_exported']}条库存风险记录。"
    )


if __name__ == "__main__":
    main()
