from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEISHU_DIR = PROJECT_ROOT / "data" / "exports" / "feishu" / "legacy_demo"
MOCK_ENTERPRISE_DIR = PROJECT_ROOT / "data" / "mock_enterprise"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"


def ensure_directories() -> None:
    for directory in (
        SYNTHETIC_DIR,
        PROCESSED_DIR,
        FEISHU_DIR,
        MOCK_ENTERPRISE_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))
