from __future__ import annotations

import csv
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "data" / "reports"
FEISHU = PROJECT_ROOT / "data" / "exports" / "feishu"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class CustomerRiskAggregationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(
            (REPORTS / "customer_risk_aggregation_audit.json").read_text(encoding="utf-8")
        )
        cls.freeze = json.loads(
            (REPORTS / "model_freeze.json").read_text(encoding="utf-8")
        )
        cls.customers = _read_csv(FEISHU / "企业渠道客户.csv")
        cls.events = _read_csv(FEISHU / "企业风险事件.csv")
        cls.tasks = _read_csv(FEISHU / "企业处置任务.csv")
        cls.evidence = _read_csv(FEISHU / "企业订单证据.csv")

    def test_open_orders_join_latest_ar_completely(self) -> None:
        self.assertEqual(self.audit["status"], "pass")
        self.assertEqual(self.audit["join_coverage"], 1.0)
        self.assertEqual(
            self.audit["open_not_overdue_orders"],
            self.audit["open_orders_joined_to_latest_ar"],
        )

    def test_prediction_and_existing_overdue_are_separate_events(self) -> None:
        counts = Counter(row["风险类型"] for row in self.events)
        predictive_count = sum(
            row["模型版本"] == self.freeze["primary_model"] for row in self.events
        )
        self.assertEqual(predictive_count, self.audit["predictive_high_risk_customers"])
        self.assertEqual(counts["存量逾期应收"], self.audit["customers_with_existing_overdue"])
        for row in self.events:
            if row["模型版本"] == self.freeze["primary_model"]:
                self.assertGreaterEqual(float(row["模型概率"]), self.freeze["risk_threshold"])
                self.assertIn("订单本身当前未逾期", row["规则解释"])
                self.assertIn("风险加权应收暴露", row)
                self.assertGreater(float(row["风险分"]), 0)
            else:
                self.assertEqual(row["模型概率"], "")
                self.assertEqual(row["模型版本"], "business_rule")
                self.assertIn("实际读取", row["规则解释"])

    def test_tasks_only_reference_red_events(self) -> None:
        event_level = {row["事件编号"]: row["风险等级"] for row in self.events}
        self.assertEqual(len(self.tasks), self.audit["red_events"])
        self.assertEqual(len({row["任务编号"] for row in self.tasks}), len(self.tasks))
        for row in self.tasks:
            expected = "TASK-" + hashlib.sha256(
                row["风险事件编号"].encode("utf-8")
            ).hexdigest()[:12].upper()
            self.assertEqual(row["任务编号"], expected)
        self.assertTrue(
            all(event_level[row["风险事件编号"]] == "红色" for row in self.tasks)
        )

    def test_feishu_exports_are_aggregated_and_bounded(self) -> None:
        self.assertGreaterEqual(len(self.customers), self.audit["latest_ar_customers"])
        evidence_counts = Counter(row["客户编号"] for row in self.evidence)
        self.assertTrue(all(count <= 3 for count in evidence_counts.values()))
        self.assertTrue(all("数据来源" in row for row in self.customers))
        self.assertTrue(all("概率加权风险金额" not in row for row in self.customers))
        self.assertTrue(
            all(
                row["客户统计口径"] == "当前有应收或近180天有采购或风险事件的客户"
                for row in self.customers
            )
        )
        event_ids = {row["事件编号"] for row in self.events}
        self.assertEqual(len({row["订单证据编号"] for row in self.evidence}), len(self.evidence))
        self.assertTrue(all(row["风险事件编号"] in event_ids for row in self.evidence))
        self.assertTrue(all(row["风险事件编号"].endswith("-PRED") for row in self.evidence))


if __name__ == "__main__":
    unittest.main()
