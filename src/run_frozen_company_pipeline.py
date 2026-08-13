from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from aggregate_company_customer_risk import aggregate_customer_risk
from audit_company_data import audit
from build_company_training_data import build_training_data
from build_inventory_health import build_inventory_health
from build_process_event_log import build_process_event_log
from build_prototype_alignment import build_prototype_alignment
from import_task_feedback import import_feedback
from project_paths import PROJECT_ROOT, portable_path
from score_frozen_payment_risk import score_frozen_model
from simulate_treatment_scenarios import simulate_scenarios


ProgressCallback = Callable[[str], None]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def run_frozen_company_pipeline(
    config_path: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    notify = progress or (lambda stage: None)
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
    model_path = models_dir / "primary_payment_risk.joblib"
    model_metrics_path = reports_dir / "model_metrics.json"
    if not model_metrics_path.exists():
        raise FileNotFoundError(f"缺少冻结模型指标：{model_metrics_path}")

    notify("审计企业数据")
    audit_report = audit(
        company_dir,
        reports_dir / "company_data_audit.json",
        docs_dir / "company_data_audit.md",
    )
    notify("构建无泄漏特征")
    training_data_report = build_training_data(
        company_dir,
        processed_dir,
        reports_dir / "training_data_audit.json",
    )
    notify("冻结模型推理")
    inference_report = score_frozen_model(
        processed_dir / "order_features.csv",
        model_path,
        reports_dir / "holdout_risk_scores.csv",
        reports_dir / "frozen_inference.json",
    )
    model_metrics = json.loads(model_metrics_path.read_text(encoding="utf-8"))
    if str(model_metrics["primary_model"]) != str(inference_report["model_type"]):
        raise ValueError("冻结模型类型与model_metrics.json不一致")
    if abs(float(model_metrics["primary_threshold"]) - float(inference_report["risk_threshold"])) > 1e-12:
        raise ValueError("冻结模型阈值与model_metrics.json不一致")
    notify("聚合客户风险")
    customer_report = aggregate_customer_risk(
        company_dir,
        processed_dir / "order_features.csv",
        reports_dir / "holdout_risk_scores.csv",
        model_metrics_path,
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
    notify("构建库存健康")
    inventory_report = build_inventory_health(
        company_dir,
        processed_dir,
        feishu_dir,
        reports_dir / "inventory_health_audit.json",
    )
    notify("生成动态到期与五维健康")
    alignment_report = build_prototype_alignment(
        company_dir,
        processed_dir / "order_features.csv",
        processed_dir,
        feishu_dir,
        reports_dir / "prototype_alignment_audit.json",
        config.get("prototype_scope", {}),
    )
    notify("生成处置情景")
    scenario_report = simulate_scenarios(
        feishu_dir / "企业渠道客户.csv",
        feishu_dir / "企业处置情景.csv",
        reports_dir / "scenario_simulation_audit.json",
    )
    notify("生成处置时间线")
    process_report = build_process_event_log(
        feishu_dir / "企业风险事件.csv",
        feishu_dir / "企业处置任务.csv",
        feishu_dir / "企业订单证据.csv",
        feishu_dir / "企业风险处置时间线.csv",
        reports_dir / "process_event_log_audit.json",
    )
    summary = {
        "status": "pass",
        "mode": "frozen_inference",
        "config": portable_path(config_path),
        "data_audit": audit_report.get("status", "pass"),
        "training_data": training_data_report["status"],
        "primary_model": inference_report["model_type"],
        "scored_orders": inference_report["scored_rows"],
        "customer_events": alignment_report["risk_events_after_alignment"],
        "inventory_risks_exported": inventory_report["feishu_export_rows"],
        "dynamic_receivable_rows": alignment_report["dynamic_monitor_rows"],
        "health_customers": alignment_report["health_customers"],
        "customer_product_exposure_rows": alignment_report["exposure_rows"],
        "scenario_rows": scenario_report["scenario_rows"],
        "process_events": process_report["event_rows"],
        "feedback_rows": int(feedback_report.get("feedback_rows", 0)),
        "customer_risk_status": customer_report["status"],
    }
    summary_path = reports_dir / "frozen_company_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = portable_path(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="使用冻结模型运行渠智罗盘企业分析主链")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "company_pipeline.json",
    )
    args = parser.parse_args()
    report = run_frozen_company_pipeline(args.config, progress=print)
    print(f"冻结推理主链完成：{report['customer_events']}条客户风险事件。")


if __name__ == "__main__":
    main()
