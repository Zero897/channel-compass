from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from feishu_client import FeishuClient
from pull_feishu_feedback import pull_feedback
from run_frozen_company_pipeline import run_frozen_company_pipeline
from sync_feishu import load_sync_config, resolve_project_path, run_sync


def build_online_runner(
    company_config_path: Path,
    sync_config_path: Path,
    client: FeishuClient,
) -> Callable[[Callable[[str], None]], dict[str, object]]:
    def runner(progress: Callable[[str], None]) -> dict[str, object]:
        company_config = json.loads(company_config_path.read_text(encoding="utf-8"))
        sync_config = load_sync_config(sync_config_path)
        reports_dir = resolve_project_path(sync_config["reports_dir"])
        feedback_path = resolve_project_path(company_config["feedback_path"])
        progress("拉取飞书人工反馈")
        pull_feedback(
            sync_config_path,
            feedback_path,
            reports_dir / "feishu_feedback_pull.json",
            client=client,
        )
        pipeline_report = run_frozen_company_pipeline(
            company_config_path,
            progress=progress,
        )
        progress("增量同步飞书")
        sync_report = run_sync(sync_config_path, apply=True, client=client)
        tables = sync_report["tables"]
        created = sum(int(table["created"]) for table in tables.values())
        updated = sum(int(table["updated"]) for table in tables.values())
        return {
            "created": created,
            "updated": updated,
            "report_path": sync_report["report_path"],
            "primary_model": pipeline_report["primary_model"],
        }

    return runner
