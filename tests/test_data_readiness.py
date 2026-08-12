from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import FEISHU_DIR, MOCK_ENTERPRISE_DIR, PROCESSED_DIR, REPORTS_DIR, read_csv  # noqa: E402
from run_84_pipeline import main  # noqa: E402


class DataReadinessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        main()

    def test_three_mock_enterprise_tables(self) -> None:
        self.assertEqual(168, len(read_csv(MOCK_ENTERPRISE_DIR / "distribution_sales.csv")))
        self.assertEqual(114, len(read_csv(MOCK_ENTERPRISE_DIR / "distribution_payments.csv")))
        self.assertEqual(72, len(read_csv(MOCK_ENTERPRISE_DIR / "distribution_ar_snapshot.csv")))

    def test_ar_snapshot_grain_and_amounts(self) -> None:
        rows = read_csv(MOCK_ENTERPRISE_DIR / "distribution_ar_snapshot.csv")
        keys = {(row["snapshot_date"], row["customer_id"]) for row in rows}
        self.assertEqual(len(rows), len(keys))
        self.assertTrue(all(float(row["receivable_balance"]) >= 0 for row in rows))
        self.assertTrue(all(float(row["overdue_balance"]) >= 0 for row in rows))

    def test_mock_data_is_clearly_labelled(self) -> None:
        for path in MOCK_ENTERPRISE_DIR.glob("*.csv"):
            rows = read_csv(path)
            self.assertTrue(rows, path.name)
            self.assertTrue(
                all(row["data_source"] == "synthetic_enterprise_mock" for row in rows),
                path.name,
            )

    def test_profile_and_validation_reports(self) -> None:
        profile = json.loads((REPORTS_DIR / "mock_data_profile.json").read_text(encoding="utf-8"))
        validation = json.loads(
            (REPORTS_DIR / "distribution_validation.json").read_text(encoding="utf-8")
        )
        self.assertEqual(3, len(profile["files"]))
        self.assertTrue(validation["valid"])
        self.assertEqual(1.0, validation["association"]["payment_customer_in_sales_coverage"])
        self.assertEqual(1.0, validation["association"]["ar_customer_in_sales_coverage"])

    def test_three_tables_regenerate_customer_risk(self) -> None:
        rows = {
            row["customer_id"]: row
            for row in read_csv(PROCESSED_DIR / "distribution_customer_features.csv")
        }
        self.assertEqual(3, len(rows))
        self.assertNotEqual("红色", rows["C001"]["overall_risk_level"])
        self.assertEqual("红色", rows["C002"]["operating_risk_level"])
        self.assertEqual("红色", rows["C003"]["receivable_risk_level"])
        self.assertTrue(
            all(
                row["threshold_status"] == "prototype_waiting_enterprise_calibration"
                for row in rows.values()
            )
        )

    def test_feishu_task_fields_separate_ai_and_human_decisions(self) -> None:
        row = read_csv(FEISHU_DIR / "处置任务.csv")[0]
        self.assertIn("AI建议动作", row)
        self.assertIn("人工处置方案", row)
        self.assertIn("执行结果", row)
        self.assertIn("预警有效性", row)
        self.assertEqual("待选择", row["人工处置方案"])
        self.assertEqual("待审批", row["审批状态"])


if __name__ == "__main__":
    unittest.main()
