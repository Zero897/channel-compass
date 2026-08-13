from __future__ import annotations

import argparse
import csv
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from feishu_client import FeishuClient
from project_paths import PROJECT_ROOT, portable_path
from sync_feishu import load_sync_config, normalize_remote_value, resolve_project_path


FEEDBACK_MAPPING = {
    "任务编号": "任务编号",
    "人工处置方案": "采用动作",
    "审批状态": "审批状态",
    "执行状态": "执行状态",
    "实际回款金额": "实际回款金额",
    "完成时间": "完成时间",
    "执行结果": "执行结果",
    "预警有效性": "预警有效性",
    "反馈备注": "备注",
}
OUTPUT_FIELDS = list(FEEDBACK_MAPPING.values())
BEIJING_TIMEZONE = timezone(timedelta(hours=8))


def _display_value(value: Any, field_type: int) -> str:
    if value is None:
        return ""
    if field_type == 5 and isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc)
        local = parsed.astimezone(BEIJING_TIMEZONE)
        if local.hour == 0 and local.minute == 0 and local.second == 0:
            return local.date().isoformat()
        return local.isoformat()
    if field_type == 2 and isinstance(value, float) and value.is_integer():
        return str(int(value))
    return normalize_remote_value(value)


def pull_feedback(
    config_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    client: FeishuClient | None = None,
) -> dict[str, Any]:
    config = load_sync_config(config_path)
    task_configs = [
        table
        for table in config["tables"]
        if table["stable_key"] == "任务编号"
        or table["local_file"] == "企业处置任务.csv"
    ]
    if len(task_configs) != 1:
        raise ValueError("飞书同步配置必须且只能包含一张处置任务表")
    task_config = task_configs[0]
    if client is None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        client = FeishuClient.from_env(
            timeout_seconds=int(config.get("request_timeout_seconds", 30)),
            env_path=str(PROJECT_ROOT / ".env"),
        )

    tables = client.list_tables()
    matching = [
        table
        for table in tables
        if str(table.get("name", "")).strip() == task_config["remote_table"]
    ]
    if len(matching) != 1:
        raise ValueError(
            f"无法唯一找到飞书处置任务表：{task_config['remote_table']}"
        )
    table_id = str(matching[0]["table_id"])
    fields = client.list_fields(table_id)
    fields_by_name = {
        str(field.get("field_name", "")): field for field in fields
    }
    missing = [name for name in FEEDBACK_MAPPING if name not in fields_by_name]
    if missing:
        raise ValueError(f"处置任务表缺少反馈字段：{', '.join(missing)}")

    records = client.search_records(table_id, field_names=FEEDBACK_MAPPING.keys())
    output_rows: list[dict[str, str]] = []
    seen_task_ids: set[str] = set()
    for record in records:
        remote_row = record.get("fields") or {}
        task_id = normalize_remote_value(remote_row.get("任务编号")).strip()
        if not task_id:
            continue
        if task_id in seen_task_ids:
            raise ValueError(f"飞书处置任务存在重复任务编号：{task_id}")
        seen_task_ids.add(task_id)
        output_row: dict[str, str] = {}
        for remote_name, output_name in FEEDBACK_MAPPING.items():
            field_type = int(fields_by_name[remote_name].get("type", 0))
            output_row[output_name] = _display_value(
                remote_row.get(remote_name), field_type
            )
        output_rows.append(output_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8-sig",
            newline="",
            delete=False,
            dir=output_path.parent,
            prefix=f".{output_path.stem}_",
            suffix=".tmp",
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(output_rows)
        Path(temporary_name).replace(output_path)
        temporary_name = ""
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()

    report = {
        "status": "pass",
        "remote_table": task_config["remote_table"],
        "records_read": len(records),
        "feedback_rows": len(output_rows),
        "output": portable_path(output_path),
        "app_token_masked": client.masked_app_token,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "反馈来自飞书人工字段；脚本不推断预警是否有效",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="从飞书处置任务拉取人工反馈")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "feishu_sync.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "feedback" / "task_feedback.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data" / "reports" / "feishu_feedback_pull.json",
    )
    args = parser.parse_args()
    report = pull_feedback(
        args.config,
        resolve_project_path(args.output),
        resolve_project_path(args.report),
    )
    print(
        f"飞书反馈拉取完成：{report['feedback_rows']}条，"
        f"输出{report['output']}"
    )


if __name__ == "__main__":
    main()
