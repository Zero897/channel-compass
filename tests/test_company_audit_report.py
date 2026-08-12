import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / "reports" / "company_data_audit.json"


class CompanyAuditReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_all_enterprise_tables_were_audited(self) -> None:
        self.assertEqual(self.report["audit_status"], "complete")
        self.assertEqual(len(self.report["tables"]), 7)
        self.assertEqual(
            sum(table["row_count"] for table in self.report["tables"].values()),
            2_652_957,
        )

    def test_snapshot_periods_match_data_description(self) -> None:
        tables = self.report["tables"]
        self.assertEqual(tables["ar_snapshots"]["distinct_snapshots"], 24)
        self.assertEqual(tables["inventory_snapshots"]["distinct_snapshots"], 8)
        self.assertEqual(tables["credit"]["distinct_customers"], 66)

    def test_core_order_relationship_is_usable(self) -> None:
        associations = self.report["associations"]
        self.assertGreaterEqual(associations["ar_orders_in_sales"]["coverage"], 0.999)
        self.assertGreaterEqual(associations["payment_orders_in_sales"]["coverage"], 0.95)

    def test_report_preserves_modeling_constraints(self) -> None:
        constraints = "\n".join(self.report["interpretation_constraints"])
        self.assertIn("不得跨24期求和", constraints)
        self.assertIn("无客户维度", constraints)


if __name__ == "__main__":
    unittest.main()
