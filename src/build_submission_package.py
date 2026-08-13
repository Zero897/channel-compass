from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from project_paths import PROJECT_ROOT, portable_path, sha256_file


REPORT_FILES = {
    "calibration_table.csv",
    "company_data_audit.json",
    "company_pipeline_summary.json",
    "customer_risk_aggregation_audit.json",
    "early_warning_evidence_audit.json",
    "feature_importance.csv",
    "frozen_company_pipeline_summary.json",
    "frozen_inference.json",
    "inventory_health_audit.json",
    "logistic_coefficients.csv",
    "model_comparison.csv",
    "model_freeze.json",
    "model_metrics.json",
    "monthly_stability.csv",
    "process_event_log_audit.json",
    "prototype_alignment_audit.json",
    "scenario_simulation_audit.json",
    "strict_early_warning_cases.csv",
    "training_data_audit.json",
}

FEISHU_TABLE_FILES = {
    "企业渠道客户.csv",
    "企业风险事件.csv",
    "企业处置任务.csv",
    "企业订单证据.csv",
    "企业处置情景.csv",
    "企业风险处置时间线.csv",
    "企业库存风险汇总.csv",
    "企业库存风险.csv",
    "企业模型指标.csv",
    "企业动态回款监控.csv",
    "企业客户商品风险暴露.csv",
}


def _package_files() -> list[Path]:
    files = [
        PROJECT_ROOT / ".env.example",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "requirements-ml.txt",
        PROJECT_ROOT / "requirements-lock.txt",
        PROJECT_ROOT / "requirements-feishu.txt",
        PROJECT_ROOT / "requirements-backend.txt",
        PROJECT_ROOT / "requirements-all.txt",
    ]
    for directory, pattern in (
        (PROJECT_ROOT / "config", "*"),
        (PROJECT_ROOT / "src", "*.py"),
        (PROJECT_ROOT / "tests", "*.py"),
        (PROJECT_ROOT / "docs", "*.md"),
        (PROJECT_ROOT / "models", "*.joblib"),
        (PROJECT_ROOT / "demo_output", "*"),
    ):
        files.extend(path for path in directory.glob(pattern) if path.is_file())
    reports = PROJECT_ROOT / "data" / "reports"
    files.extend(reports / name for name in REPORT_FILES)
    feishu = PROJECT_ROOT / "data" / "exports" / "feishu"
    files.extend(feishu / name for name in FEISHU_TABLE_FILES)
    files.append(PROJECT_ROOT / "data" / "feedback" / "task_feedback_template.csv")
    missing = [portable_path(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"提交包缺少文件：{', '.join(missing)}")
    return sorted(set(files), key=portable_path)


def build_package(output_path: Path) -> dict[str, object]:
    files = _package_files()
    manifest_files = [
        {
            "path": portable_path(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "files": manifest_files,
        "excluded_large_artifacts": [
            ".env",
            "data/feedback/task_feedback.csv",
            "data/reports/holdout_risk_scores.csv",
            "data/reports/test_predictions.csv",
            "data/processed/order_features.csv",
            "data/company/AFFT模拟数据集",
        ],
        "reproduction": {
            "create_environment": "python -m venv .venv",
            "install": "& .\\.venv\\Scripts\\python.exe -m pip install -r requirements-all.txt",
            "portable_demo": "& .\\.venv\\Scripts\\python.exe src\\run_portable_demo.py",
            "public_tests": "& .\\.venv\\Scripts\\python.exe -m unittest tests.test_pipeline tests.test_portable_demo tests.test_backend_api tests.test_feishu_sync tests.test_run_control_store tests.test_run_service -v",
            "frozen_enterprise_inference": "补齐data/company/AFFT模拟数据集后运行：& .\\.venv\\Scripts\\python.exe src\\run_frozen_company_pipeline.py",
            "administrator_retraining": "仅重新训练和冻结模型时运行：& .\\.venv\\Scripts\\python.exe src\\run_company_pipeline.py",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, portable_path(path))
        archive.writestr(
            "PACKAGE_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    return {
        "status": "pass",
        "output": portable_path(output_path),
        "file_count": len(files) + 1,
        "size_bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="构建不含企业原始数据和大文件的渠智罗盘提交包")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dist" / "渠智罗盘_比赛提交包_20260813.zip",
    )
    args = parser.parse_args()
    result = build_package(args.output)
    print(
        f"提交包完成：{result['file_count']}个文件，{result['size_bytes'] / 1024 / 1024:.2f}MB，"
        f"SHA256={result['sha256']}"
    )


if __name__ == "__main__":
    main()
