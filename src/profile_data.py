from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from project_paths import portable_path


UNIQUE_SAMPLE_LIMIT = 100_000
TYPE_SAMPLE_LIMIT = 20_000


def _detect_encoding(path: Path) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding) as handle:
                handle.read(65_536)
            return encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"无法识别文件编码：{path}")


def _classify(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return "empty"
    try:
        float(stripped.replace(",", ""))
        return "number"
    except ValueError:
        pass
    try:
        datetime.fromisoformat(stripped.replace("Z", "+00:00"))
        return "date"
    except ValueError:
        return "text"


def profile_csv(path: Path) -> dict[str, object]:
    encoding = _detect_encoding(path)
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV没有表头：{path}")
        fields = reader.fieldnames
        missing = Counter({field: 0 for field in fields})
        type_counts = {field: Counter() for field in fields}
        unique_samples: dict[str, set[str]] = {field: set() for field in fields}
        row_count = 0
        for row in reader:
            row_count += 1
            for field in fields:
                value = (row.get(field) or "").strip()
                if not value:
                    missing[field] += 1
                if row_count <= TYPE_SAMPLE_LIMIT:
                    type_counts[field][_classify(value)] += 1
                if len(unique_samples[field]) < UNIQUE_SAMPLE_LIMIT:
                    unique_samples[field].add(value)

    column_profiles = []
    type_sample_count = min(row_count, TYPE_SAMPLE_LIMIT)
    for field in fields:
        non_empty_types = {
            key: count for key, count in type_counts[field].items() if key != "empty"
        }
        inferred_type = max(non_empty_types, key=non_empty_types.get) if non_empty_types else "empty"
        column_profiles.append(
            {
                "column": field,
                "inferred_type": inferred_type,
                "missing_count": missing[field],
                "missing_ratio": round(missing[field] / row_count, 6) if row_count else 0,
                "sample_unique_count": len(unique_samples[field]),
                "uniqueness_sample_capped": len(unique_samples[field]) >= UNIQUE_SAMPLE_LIMIT,
                "type_sample_count": type_sample_count,
            }
        )
    return {
        "file": path.name,
        "file_size_bytes": path.stat().st_size,
        "encoding": encoding,
        "row_count": row_count,
        "column_count": len(fields),
        "columns": column_profiles,
    }


def profile_directory(input_dir: Path, output_path: Path) -> dict[str, object]:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"目录中没有CSV文件：{input_dir}")
    report = {
        "input_directory": portable_path(input_dir),
        "note": "行数和缺失数为全量统计；类型和唯一值为有上限的样本统计。",
        "files": [profile_csv(path) for path in files],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="生成CSV数据体检报告")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = profile_directory(args.input_dir, args.output)
    print(f"完成数据体检：{len(report['files'])}个CSV文件。")


if __name__ == "__main__":
    main()
