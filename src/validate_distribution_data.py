from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path


KEY_SAMPLE_LIMIT = 250_000


def _load_config(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_date_field(logical_name: str) -> bool:
    return logical_name.endswith("_date")


def _is_number_field(logical_name: str) -> bool:
    return logical_name.endswith("_amount") or logical_name.endswith("_balance")


def _table_key(logical_table: str, logical_row: dict[str, str]) -> str:
    if logical_table == "sales":
        return logical_row.get("sale_id", "")
    if logical_table == "payments":
        return logical_row.get("payment_id", "")
    return f"{logical_row.get('snapshot_date', '')}|{logical_row.get('customer_id', '')}"


def validate(
    input_dir: Path,
    config_path: Path,
    output_path: Path,
) -> dict[str, object]:
    config = _load_config(config_path)
    tables = config["tables"]
    reports: dict[str, dict[str, object]] = {}
    customer_sets: dict[str, set[str]] = {}
    overall_valid = True

    for logical_table, table_config in tables.items():
        file_path = input_dir / table_config["file"]
        if not file_path.exists():
            reports[logical_table] = {"valid": False, "error": "file_not_found"}
            overall_valid = False
            continue
        with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = set(reader.fieldnames or [])
            field_mapping: dict[str, str] = table_config["fields"]
            required_logical: list[str] = table_config["required"]
            missing_headers = [
                logical
                for logical in required_logical
                if field_mapping.get(logical) not in headers
            ]
            row_count = 0
            empty_required = 0
            invalid_dates = 0
            invalid_numbers = 0
            negative_amounts = 0
            duplicate_key_sample = 0
            seen_keys: set[str] = set()
            customer_ids: set[str] = set()
            for raw_row in reader:
                row_count += 1
                logical_row = {
                    logical: (raw_row.get(raw_name) or "").strip()
                    for logical, raw_name in field_mapping.items()
                }
                if any(not logical_row.get(field, "") for field in required_logical):
                    empty_required += 1
                customer_id = logical_row.get("customer_id", "")
                if customer_id:
                    customer_ids.add(customer_id)
                for logical, value in logical_row.items():
                    if not value:
                        continue
                    if _is_date_field(logical):
                        try:
                            date.fromisoformat(value[:10])
                        except ValueError:
                            invalid_dates += 1
                    elif _is_number_field(logical):
                        try:
                            number = float(value.replace(",", ""))
                            if number < 0:
                                negative_amounts += 1
                        except ValueError:
                            invalid_numbers += 1
                key = _table_key(logical_table, logical_row)
                if key and len(seen_keys) < KEY_SAMPLE_LIMIT:
                    if key in seen_keys:
                        duplicate_key_sample += 1
                    seen_keys.add(key)

        table_valid = (
            not missing_headers
            and row_count > 0
            and empty_required == 0
            and invalid_dates == 0
            and invalid_numbers == 0
            and duplicate_key_sample == 0
        )
        reports[logical_table] = {
            "valid": table_valid,
            "file": file_path.name,
            "row_count": row_count,
            "missing_required_headers": missing_headers,
            "rows_with_empty_required": empty_required,
            "invalid_date_values": invalid_dates,
            "invalid_number_values": invalid_numbers,
            "negative_amount_values": negative_amounts,
            "duplicate_key_count_in_sample": duplicate_key_sample,
            "key_sample_capped": len(seen_keys) >= KEY_SAMPLE_LIMIT,
            "customer_count": len(customer_ids),
        }
        customer_sets[logical_table] = customer_ids
        overall_valid = overall_valid and table_valid

    sales_customers = customer_sets.get("sales", set())
    payment_customers = customer_sets.get("payments", set())
    ar_customers = customer_sets.get("ar_snapshots", set())
    payment_coverage = (
        len(payment_customers & sales_customers) / len(payment_customers)
        if payment_customers
        else 0.0
    )
    ar_coverage = (
        len(ar_customers & sales_customers) / len(ar_customers) if ar_customers else 0.0
    )
    association = {
        "payment_customer_in_sales_coverage": round(payment_coverage, 6),
        "ar_customer_in_sales_coverage": round(ar_coverage, 6),
        "payment_customers_not_in_sales": sorted(payment_customers - sales_customers)[:20],
        "ar_customers_not_in_sales": sorted(ar_customers - sales_customers)[:20],
    }
    overall_valid = overall_valid and payment_coverage == 1.0 and ar_coverage == 1.0
    report = {
        "valid": overall_valid,
        "business_type": config["business_type"],
        "config_version": config["version"],
        "tables": reports,
        "association": association,
        "limitations": [
            "企业字段名和业务口径尚未确认",
            "超过25万条时主键重复检查采用前25万条样本",
            "负金额只统计不直接判错，需结合退货和冲销口径解释",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="校验非项目类销售、回款、应收三表")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = validate(args.input_dir, args.config, args.output)
    print("三表校验通过。" if report["valid"] else "三表校验未通过，请查看报告。")
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
