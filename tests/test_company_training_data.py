from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_company_training_data import (  # noqa: E402
    FEATURE_FIELDS,
    _identifier,
    _label,
    _payment_term_days,
    _split,
)


class CompanyTrainingHelpersTest(unittest.TestCase):
    def test_identifier_and_payment_term_normalization(self) -> None:
        self.assertEqual(_identifier("3053157715.0"), "3053157715")
        self.assertEqual(_identifier("LXBJF25110256"), "LXBJF25110256")
        self.assertEqual(_payment_term_days("95 天付款"), 95)
        self.assertEqual(_payment_term_days("立即付款"), 0)

    def test_time_split_boundaries(self) -> None:
        self.assertEqual(_split("2025-03-31"), "train")
        self.assertEqual(_split("2025-04-01"), "embargo")
        self.assertEqual(_split("2025-08-01"), "validation")
        self.assertEqual(_split("2025-10-01"), "embargo")
        self.assertEqual(_split("2026-02-01"), "test")
        self.assertEqual(_split("2026-04-01"), "scoring_holdout")

    def test_label_requires_observed_outcome(self) -> None:
        base = {
            "is_project": 0,
            "outcome_snapshot_date": "2025-05-31",
            "ever_overdue_payment": 0,
            "ever_overdue_ar": 0,
            "first_overdue_payment_date": None,
            "first_overdue_ar_date": None,
            "has_payment": 1,
            "present_at_outcome_snapshot": 0,
            "present_in_latest_ar": 0,
            "last_payment_date": "2025-01-10",
        }
        self.assertEqual(_label(base), ("0", "eligible", "2025-05-31"))
        self.assertEqual(
            _label({**base, "ever_overdue_payment": 1, "first_overdue_payment_date": "2025-01-08"}),
            ("1", "eligible", "2025-01-08"),
        )
        self.assertEqual(
            _label({**base, "has_payment": 0, "present_at_outcome_snapshot": 1}),
            ("", "insufficient_open_receivable", ""),
        )
        self.assertEqual(
            _label({**base, "outcome_snapshot_date": None}),
            ("", "insufficient_unmatured", ""),
        )


class CompanyTrainingOutputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        report_path = PROJECT_ROOT / "data" / "reports" / "training_data_audit.json"
        if not report_path.exists():
            raise RuntimeError("请先运行 build_company_training_data.py 生成训练数据")
        cls.report = json.loads(report_path.read_text(encoding="utf-8"))

    def test_leakage_gate_passes(self) -> None:
        self.assertEqual(self.report["status"], "pass")
        self.assertEqual(self.report["leakage_check"], "pass")
        self.assertEqual(self.report["forbidden_feature_intersection"], [])

    def test_each_evaluation_split_has_both_classes(self) -> None:
        for split in ("train", "validation", "test"):
            with self.subTest(split=split):
                stats = self.report["summary"]["splits"][split]
                self.assertGreater(stats["positive"], 0)
                self.assertGreater(stats["negative"], 0)

    def test_feature_header_is_frozen(self) -> None:
        feature_path = PROJECT_ROOT / "data" / "processed" / "order_features.csv"
        if not feature_path.exists():
            self.skipTest("提交包不包含完整企业训练特征；放入数据并运行企业主链后执行")
        with feature_path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
        self.assertEqual(header, FEATURE_FIELDS)

    def test_status_counts_cover_every_order(self) -> None:
        summary = self.report["summary"]
        self.assertEqual(sum(summary["label_status_counts"].values()), summary["total_orders"])


if __name__ == "__main__":
    unittest.main()
