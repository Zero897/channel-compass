from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import FEISHU_DIR, PROCESSED_DIR, SYNTHETIC_DIR, read_csv  # noqa: E402
from run_pipeline import main  # noqa: E402


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        main()

    def test_seven_business_tables_exist_and_are_synthetic(self) -> None:
        filenames = {
            "customer.csv",
            "product.csv",
            "sales_order.csv",
            "inventory_snapshot.csv",
            "invoice.csv",
            "payment.csv",
            "intervention.csv",
        }
        self.assertEqual(filenames, {path.name for path in SYNTHETIC_DIR.glob("*.csv")})
        for filename in filenames:
            rows = read_csv(SYNTHETIC_DIR / filename)
            self.assertTrue(rows, filename)
            self.assertTrue(all(row["data_source"] == "synthetic" for row in rows), filename)

    def test_demo_scope_and_feishu_exports(self) -> None:
        self.assertEqual(3, len(read_csv(SYNTHETIC_DIR / "customer.csv")))
        self.assertEqual(5, len(read_csv(SYNTHETIC_DIR / "product.csv")))
        self.assertEqual(10, len(read_csv(FEISHU_DIR / "风险事件.csv")))
        expected = {"渠道客户.csv", "风险事件.csv", "处置任务.csv", "模型指标.csv"}
        self.assertTrue(expected.issubset({path.name for path in FEISHU_DIR.glob("*.csv")}))

    def test_planted_cases_are_identified(self) -> None:
        rows = {row["customer_id"]: row for row in read_csv(PROCESSED_DIR / "customer_features.csv")}
        self.assertEqual("黄色", rows["C001"]["overall_risk_level"])
        self.assertNotEqual("红色", rows["C001"]["overall_risk_level"])
        self.assertEqual("红色", rows["C002"]["inventory_risk_level"])
        self.assertEqual("红色", rows["C003"]["inventory_risk_level"])
        self.assertEqual("红色", rows["C003"]["payment_risk_level"])
        self.assertEqual("红色", rows["C003"]["overall_risk_level"])

    def test_no_scenario_label_leaks_into_feature_table(self) -> None:
        headers = set(read_csv(PROCESSED_DIR / "customer_features.csv")[0])
        self.assertFalse({"scenario", "case_type", "expected_label"} & headers)

    def test_c003_local_flow_reaches_completed_state(self) -> None:
        intervention = read_csv(SYNTHETIC_DIR / "intervention.csv")
        self.assertEqual("C003", intervention[0]["customer_id"])
        self.assertEqual("已完成", intervention[0]["status"])
        trace = json.loads((PROCESSED_DIR / "c003_demo_trace.json").read_text(encoding="utf-8"))
        self.assertEqual(["待处理", "处理中", "已完成"], [row["task_status"] for row in trace[2:]])


if __name__ == "__main__":
    unittest.main()
