from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from feishu_client import FeishuClient
from project_paths import PROJECT_ROOT, portable_path


BEIJING_TIMEZONE = timezone(timedelta(hours=8))
READ_ONLY_FIELD_TYPES = {11, 18, 19, 20, 21, 23, 24, 1001, 1002, 1003, 1004, 1005, 3001}


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_sync_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"source_dir", "reports_dir", "batch_size", "tables"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"飞书同步配置缺少字段：{', '.join(missing)}")
    batch_size = int(config["batch_size"])
    if not 1 <= batch_size <= 1000:
        raise ValueError("batch_size必须在1到1000之间")
    tables = config["tables"]
    if not isinstance(tables, list) or not tables:
        raise ValueError("tables必须是非空数组")
    remote_names: set[str] = set()
    local_files: set[str] = set()
    for table in tables:
        for name in ("local_file", "remote_table", "stable_key", "update_fields"):
            if name not in table:
                raise ValueError(f"表配置缺少{name}：{table}")
        remote_name = str(table["remote_table"]).strip()
        local_file = str(table["local_file"]).strip()
        if remote_name in remote_names:
            raise ValueError(f"重复远端表名：{remote_name}")
        if local_file in local_files:
            raise ValueError(f"重复本地文件：{local_file}")
        remote_names.add(remote_name)
        local_files.add(local_file)
        update_fields = set(table["update_fields"])
        create_only_fields = set(table.get("create_only_fields", []))
        overlap_modes = sorted(update_fields & create_only_fields)
        if overlap_modes:
            raise ValueError(
                f"{remote_name}的可更新字段与仅创建字段冲突：{', '.join(overlap_modes)}"
            )
        field_mappings = table.get("field_mappings", {})
        if not isinstance(field_mappings, dict):
            raise ValueError(f"{remote_name}的field_mappings必须是对象")
        writable_local_fields = update_fields | create_only_fields
        unknown_mapping_fields = sorted(set(field_mappings) - writable_local_fields)
        if unknown_mapping_fields:
            raise ValueError(
                f"{remote_name}的字段映射来源不在可更新字段中："
                f"{', '.join(unknown_mapping_fields)}"
            )
        remote_update_fields = [
            str(field_mappings.get(field_name, field_name)).strip()
            for field_name in table["update_fields"]
        ]
        remote_create_only_fields = [
            str(field_mappings.get(field_name, field_name)).strip()
            for field_name in table.get("create_only_fields", [])
        ]
        remote_writable_fields = [*remote_update_fields, *remote_create_only_fields]
        if any(not field_name for field_name in remote_writable_fields):
            raise ValueError(f"{remote_name}的字段映射目标不能为空")
        if len(remote_writable_fields) != len(set(remote_writable_fields)):
            raise ValueError(f"{remote_name}的字段映射目标存在重复")
        protected_fields = set(table.get("protected_fields", []))
        overlap = sorted(set(remote_writable_fields) & protected_fields)
        if overlap:
            raise ValueError(
                f"{remote_name}的可更新字段与保护字段冲突：{', '.join(overlap)}"
            )
        if table["stable_key"] in protected_fields:
            raise ValueError(f"{remote_name}的稳定键不能是保护字段")
    return config


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"缺少同步源文件：{portable_path(path)}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV没有表头：{portable_path(path)}")
        rows = [{key: value or "" for key, value in row.items()} for row in reader]
        return list(reader.fieldnames), rows


def validate_stable_ids(
    rows: list[dict[str, Any]], stable_key: str, source_name: str
) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    empty_rows: list[int] = []
    duplicates: list[str] = []
    for index, row in enumerate(rows, start=2):
        value = normalize_remote_value(row.get(stable_key)).strip()
        if not value:
            empty_rows.append(index)
            continue
        if value in mapping:
            duplicates.append(value)
            continue
        mapping[value] = row
    if empty_rows:
        raise ValueError(
            f"{source_name}的{stable_key}存在空值，首个CSV/记录行：{empty_rows[0]}"
        )
    if duplicates:
        unique = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(f"{source_name}的{stable_key}存在重复：{unique}")
    return mapping


def normalize_remote_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return ""
        return str(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "name" in item:
                    parts.append(str(item["name"]))
                elif "id" in item:
                    parts.append(str(item["id"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(value, dict):
        if "text" in value:
            return str(value["text"])
        if "link_record_ids" in value:
            return ",".join(str(item) for item in value["link_record_ids"])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _parse_number(raw: Any, field_name: str) -> int | float:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        return raw
    text = str(raw).strip().replace(",", "").replace("¥", "").replace("￥", "")
    if text.endswith("%"):
        return float(text[:-1]) / 100
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{field_name}不是有效数字：{raw}") from exc
    return int(value) if value.is_integer() else value


def _parse_date_milliseconds(raw: Any, field_name: str) -> int:
    if isinstance(raw, (int, float)):
        return int(raw)
    text = str(raw).strip()
    formats = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%Y/%m/%d %H:%M:%S")
    parsed: datetime | None = None
    for date_format in formats:
        try:
            parsed = datetime.strptime(text, date_format)
            break
        except ValueError:
            continue
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name}不是有效日期：{raw}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=BEIJING_TIMEZONE)
    return int(parsed.timestamp() * 1000)


def convert_for_feishu(raw: Any, field: dict[str, Any]) -> Any:
    field_name = str(field.get("field_name", "未知字段"))
    field_type = int(field.get("type", 0))
    if _is_blank(raw):
        return None
    if field_type == 1:
        return str(raw)
    if field_type == 2:
        return _parse_number(raw, field_name)
    if field_type == 3:
        return str(raw).strip()
    if field_type == 4:
        if isinstance(raw, list):
            return raw
        return [item.strip() for item in str(raw).split(",") if item.strip()]
    if field_type == 5:
        return _parse_date_milliseconds(raw, field_name)
    if field_type == 7:
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in {"true", "1", "是", "yes"}:
            return True
        if text in {"false", "0", "否", "no"}:
            return False
        raise ValueError(f"{field_name}不是有效复选框值：{raw}")
    if field_type == 13:
        return str(raw).strip()
    if field_type == 15:
        if isinstance(raw, dict):
            return raw
        return {"text": str(raw), "link": str(raw)}
    if field_type in {18, 21}:
        if not isinstance(raw, dict) or "link_record_ids" not in raw:
            raise ValueError(f"{field_name}关联字段必须使用link_record_ids对象")
        return raw
    raise ValueError(f"{field_name}的飞书字段类型{field_type}不允许由同步程序写入")


def values_equal(local: Any, remote: Any, field_type: int) -> bool:
    if field_type == 2:
        try:
            return math.isclose(float(local), float(remote), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    if field_type == 5:
        try:
            return int(local) == int(remote)
        except (TypeError, ValueError):
            return False
    if field_type == 7:
        if isinstance(remote, str):
            remote = remote.lower() in {"true", "1", "是", "yes"}
        return bool(local) == bool(remote)
    return normalize_remote_value(local) == normalize_remote_value(remote)


def _field_map(fields: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for field in fields:
        name = str(field.get("field_name", "")).strip()
        if not name:
            continue
        if name in mapping:
            raise ValueError(f"飞书存在重复字段名：{name}")
        mapping[name] = field
    return mapping


def _validate_table_contract(
    table_config: dict[str, Any],
    local_headers: list[str],
    remote_fields: dict[str, dict[str, Any]],
) -> None:
    table_name = table_config["remote_table"]
    stable_key = table_config["stable_key"]
    required_local = [
        stable_key,
        *table_config["update_fields"],
        *table_config.get("create_only_fields", []),
    ]
    missing_local = [name for name in required_local if name not in local_headers]
    if missing_local:
        raise ValueError(f"{table_name}本地CSV缺少字段：{', '.join(missing_local)}")
    field_mappings = table_config.get("field_mappings", {})
    required_remote = [
        stable_key,
        *[
            field_mappings.get(field_name, field_name)
            for field_name in table_config["update_fields"]
        ],
        *[
            field_mappings.get(field_name, field_name)
            for field_name in table_config.get("create_only_fields", [])
        ],
        *table_config.get("create_defaults", {}).keys(),
    ]
    missing_remote = [name for name in required_remote if name not in remote_fields]
    if missing_remote:
        raise ValueError(f"{table_name}飞书表缺少字段：{', '.join(missing_remote)}")
    for field_name in required_remote:
        field_type = int(remote_fields[field_name].get("type", 0))
        if field_name == stable_key and field_type == 1:
            continue
        if field_type in READ_ONLY_FIELD_TYPES:
            raise ValueError(
                f"{table_name}.{field_name}为只读或关联类型{field_type}，不能进入普通同步白名单"
            )


def build_upsert_plan(
    table_config: dict[str, Any],
    local_rows: list[dict[str, str]],
    remote_records: list[dict[str, Any]],
    fields: list[dict[str, Any]],
) -> dict[str, Any]:
    table_name = table_config["remote_table"]
    stable_key = table_config["stable_key"]
    remote_fields = _field_map(fields)
    field_mappings = table_config.get("field_mappings", {})
    local_by_id = validate_stable_ids(local_rows, stable_key, table_config["local_file"])
    remote_rows: list[dict[str, Any]] = []
    record_ids: dict[str, str] = {}
    remote_missing_stable_key = 0
    for record in remote_records:
        record_id = str(record.get("record_id", "")).strip()
        row = record.get("fields") or {}
        stable_id = normalize_remote_value(row.get(stable_key)).strip()
        if not stable_id:
            remote_missing_stable_key += 1
            continue
        remote_rows.append(row)
        if not record_id:
            raise ValueError(f"{table_name}记录{stable_id}缺少record_id")
        record_ids[stable_id] = record_id
    remote_by_id = validate_stable_ids(remote_rows, stable_key, table_name)
    creates: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    skipped = 0

    for stable_id, local_row in local_by_id.items():
        is_new = stable_id not in remote_by_id
        candidate_fields: dict[str, Any] = {}
        create_only_fields = set(table_config.get("create_only_fields", []))
        fields_to_write = [
            stable_key,
            *table_config["update_fields"],
            *table_config.get("create_only_fields", []),
        ]
        for field_name in fields_to_write:
            if not is_new and field_name in create_only_fields:
                continue
            remote_field_name = (
                stable_key
                if field_name == stable_key
                else field_mappings.get(field_name, field_name)
            )
            raw = local_row.get(field_name, "")
            converted = convert_for_feishu(raw, remote_fields[remote_field_name])
            if converted is None:
                continue
            if is_new or field_name != stable_key:
                candidate_fields[remote_field_name] = converted
        if is_new:
            for field_name, default in table_config.get("create_defaults", {}).items():
                converted = convert_for_feishu(default, remote_fields[field_name])
                if converted is not None:
                    candidate_fields[field_name] = converted
            creates.append({"fields": candidate_fields})
            continue

        changed_fields: dict[str, Any] = {}
        remote_row = remote_by_id[stable_id]
        for field_name, local_value in candidate_fields.items():
            if field_name == stable_key:
                continue
            field_type = int(remote_fields[field_name].get("type", 0))
            if not values_equal(local_value, remote_row.get(field_name), field_type):
                changed_fields[field_name] = local_value
        if changed_fields:
            updates.append(
                {"record_id": record_ids[stable_id], "fields": changed_fields}
            )
        else:
            skipped += 1

    return {
        "local_rows": len(local_rows),
        "remote_rows_before": len(remote_records),
        "creates": creates,
        "updates": updates,
        "skipped": skipped,
        "remote_only": len(set(remote_by_id) - set(local_by_id)),
        "remote_missing_stable_key": remote_missing_stable_key,
    }


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _select_tables(
    tables: list[dict[str, Any]], selected_tables: set[str] | None
) -> list[dict[str, Any]]:
    if not selected_tables:
        return tables
    selected: list[dict[str, Any]] = []
    matched: set[str] = set()
    for table in tables:
        aliases = {
            str(table["remote_table"]),
            str(table["local_file"]),
            Path(str(table["local_file"])).stem,
        }
        hits = aliases & selected_tables
        if hits:
            selected.append(table)
            matched.update(hits)
    unknown = sorted(selected_tables - matched)
    if unknown:
        raise ValueError(f"--tables包含未知表：{', '.join(unknown)}")
    return selected


def run_sync(
    config_path: Path,
    *,
    apply: bool,
    selected_tables: set[str] | None = None,
    client: FeishuClient | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    config = load_sync_config(config_path)
    source_dir = resolve_project_path(config["source_dir"])
    reports_dir = resolve_project_path(config["reports_dir"])
    tables = _select_tables(config["tables"], selected_tables)
    if not tables:
        raise ValueError("没有需要同步的数据表")
    if client is None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        if apply and os.getenv("FEISHU_DRY_RUN", "true").strip().lower() != "false":
            raise ValueError("FEISHU_DRY_RUN仍为true；正式写入前请显式改为false")
        client = FeishuClient.from_env(
            timeout_seconds=int(config.get("request_timeout_seconds", 30)),
            env_path=str(PROJECT_ROOT / ".env"),
        )

    discovered = client.list_tables()
    tables_by_name: dict[str, str] = {}
    for item in discovered:
        name = str(item.get("name", "")).strip()
        table_id = str(item.get("table_id", "")).strip()
        if not name or not table_id:
            continue
        if name in tables_by_name:
            raise ValueError(f"飞书存在重名数据表：{name}")
        tables_by_name[name] = table_id

    missing_tables = [
        table["remote_table"]
        for table in tables
        if table["remote_table"] not in tables_by_name
    ]
    if missing_tables:
        available = ", ".join(sorted(tables_by_name))
        raise ValueError(
            f"飞书缺少配置表：{', '.join(missing_tables)}；实际发现：{available}"
        )

    report: dict[str, Any] = {
        "mode": "apply" if apply else "dry-run",
        "started_at": started_at.isoformat(),
        "finished_at": "",
        "config": portable_path(config_path),
        "app_token_masked": client.masked_app_token,
        "tables": {},
        "protected_field_overwrite_attempts": 0,
        "status": "pass",
    }
    batch_size = int(config["batch_size"])

    try:
        for table_config in tables:
            table_name = table_config["remote_table"]
            table_id = tables_by_name[table_name]
            local_path = source_dir / table_config["local_file"]
            headers, local_rows = read_csv_rows(local_path)
            fields = client.list_fields(table_id)
            field_names = [str(field.get("field_name", "")) for field in fields]
            _validate_table_contract(table_config, headers, _field_map(fields))
            remote_records = client.search_records(table_id, field_names=field_names)
            plan = build_upsert_plan(
                table_config, local_rows, remote_records, fields
            )
            created = 0
            updated = 0
            if apply:
                for batch in _chunks(plan["creates"], batch_size):
                    created += len(client.batch_create(table_id, batch))
                for batch in _chunks(plan["updates"], batch_size):
                    updated += len(client.batch_update(table_id, batch))
            table_report = {
                "local_rows": plan["local_rows"],
                "remote_rows_before": plan["remote_rows_before"],
                "planned_create": len(plan["creates"]),
                "planned_update": len(plan["updates"]),
                "created": created,
                "updated": updated,
                "skipped": plan["skipped"],
                "remote_only": plan["remote_only"],
                "remote_missing_stable_key": plan["remote_missing_stable_key"],
                "failed": 0,
            }
            report["tables"][table_name] = table_report
            print(
                f"{table_name}: 新增{table_report['planned_create']}，"
                f"更新{table_report['planned_update']}，跳过{table_report['skipped']}"
                f"，远端缺少{table_config['stable_key']}"
                f"{table_report['remote_missing_stable_key']}"
            )
    except Exception as exc:
        report["status"] = "fail"
        report["error"] = str(exc)
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = started_at.strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"feishu_sync_{timestamp}.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["report_path"] = portable_path(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="增量同步渠智罗盘CSV到飞书多维表格")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "feishu_sync.json",
    )
    parser.add_argument(
        "--tables",
        default="",
        help="逗号分隔的远端表名、本地文件名或文件主名；留空表示全部配置表",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只生成同步计划")
    mode.add_argument("--apply", action="store_true", help="执行新增和更新")
    args = parser.parse_args()
    selected = {item.strip() for item in args.tables.split(",") if item.strip()} or None
    report = run_sync(
        args.config,
        apply=bool(args.apply),
        selected_tables=selected,
    )
    print(
        f"飞书同步{report['status']}（{report['mode']}），报告：{report['report_path']}"
    )


if __name__ == "__main__":
    main()
