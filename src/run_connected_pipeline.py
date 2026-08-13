from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from feishu_client import FeishuClient
from project_paths import PROJECT_ROOT, portable_path
from pull_feishu_feedback import pull_feedback
from run_company_pipeline import run_company_pipeline
from sync_feishu import load_sync_config, resolve_project_path, run_sync


def run_connected_pipeline(
    company_config_path: Path,
    sync_config_path: Path,
    *,
    apply: bool,
    client: FeishuClient | None = None,
) -> dict[str, Any]:
    company_config = json.loads(company_config_path.read_text(encoding="utf-8"))
    sync_config = load_sync_config(sync_config_path)
    reports_dir = resolve_project_path(company_config["reports_dir"])
    feedback_path = resolve_project_path(company_config["feedback_path"])
    feedback_report_path = reports_dir / "feishu_feedback_pull.json"
    if client is None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        if apply and os.getenv("FEISHU_DRY_RUN", "true").strip().lower() != "false":
            raise ValueError("FEISHU_DRY_RUN仍为true；正式写入前请显式改为false")
        client = FeishuClient.from_env(
            timeout_seconds=int(sync_config.get("request_timeout_seconds", 30)),
            env_path=str(PROJECT_ROOT / ".env"),
        )

    print("[连接1/3] 从飞书拉取人工反馈")
    feedback_report = pull_feedback(
        sync_config_path,
        feedback_path,
        feedback_report_path,
        client=client,
    )
    print("[连接2/3] 运行企业分析主链")
    company_report = run_company_pipeline(company_config_path)
    print("[连接3/3] 增量同步分析结果")
    sync_report = run_sync(
        sync_config_path,
        apply=apply,
        client=client,
    )
    summary = {
        "status": "pass",
        "mode": "apply" if apply else "dry-run",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "company_config": portable_path(company_config_path),
        "sync_config": portable_path(sync_config_path),
        "feedback_rows": feedback_report["feedback_rows"],
        "primary_model": company_report["primary_model"],
        "risk_events": company_report["customer_events"],
        "sync_report": sync_report["report_path"],
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "connected_pipeline_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["summary_path"] = portable_path(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="运行渠智罗盘企业主链并同步飞书")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "company_pipeline.json",
    )
    parser.add_argument(
        "--sync-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "feishu_sync.json",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="运行主链但不写飞书")
    mode.add_argument("--apply", action="store_true", help="运行主链并写入飞书")
    args = parser.parse_args()
    summary = run_connected_pipeline(
        args.config,
        args.sync_config,
        apply=bool(args.apply),
    )
    print(
        f"连接主链完成（{summary['mode']}）：{summary['summary_path']}"
    )


if __name__ == "__main__":
    main()
