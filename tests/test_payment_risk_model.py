from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "data" / "reports"
MODELS = PROJECT_ROOT / "models"


class PaymentRiskModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metrics = json.loads(
            (REPORTS / "model_metrics.json").read_text(encoding="utf-8")
        )
        cls.freeze = json.loads(
            (REPORTS / "model_freeze.json").read_text(encoding="utf-8")
        )

    def test_primary_model_is_frozen_consistently(self) -> None:
        self.assertTrue(self.metrics["frozen"])
        self.assertEqual(self.freeze["status"], "frozen")
        self.assertEqual(self.metrics["primary_model"], self.freeze["primary_model"])
        self.assertEqual(self.metrics["primary_threshold"], self.freeze["risk_threshold"])
        self.assertTrue((MODELS / "primary_payment_risk.joblib").exists())
        self.assertEqual(self.freeze["selection_split"], "validation")
        candidates = [
            item
            for item in self.metrics["models"]
            if item["model"] in {"logistic_regression", "lightgbm"}
        ]
        expected = max(
            candidates,
            key=lambda item: (
                item["splits"]["validation"]["pr_auc"],
                item["splits"]["validation"]["risk_amount_capture"],
            ),
        )["model"]
        self.assertEqual(expected, self.metrics["primary_model"])

    def test_logistic_regression_converged(self) -> None:
        logistic = next(
            item for item in self.metrics["models"] if item["model"] == "logistic_regression"
        )
        training = logistic["training"]
        self.assertEqual(training["solver"], "lbfgs")
        self.assertTrue(training["converged"])
        self.assertLess(training["actual_iterations"], training["max_iterations"])

    def test_primary_model_beats_rule_on_time_test(self) -> None:
        results = {
            item["model"]: item["splits"]["test"] for item in self.metrics["models"]
        }
        primary = results[self.metrics["primary_model"]]
        self.assertGreater(primary["pr_auc"], results["business_rule"]["pr_auc"])
        self.assertGreater(primary["pr_auc"], primary["positive_rate"])
        self.assertGreater(primary["risk_amount_capture"], 0.5)
        self.assertIn("top20_risk_amount_capture", primary)
        self.assertIn("business_false_alarm_rate", primary)

    def test_simple_baseline_and_ablation_are_reported(self) -> None:
        results = {item["model"]: item for item in self.metrics["models"]}
        self.assertIn("history_overdue_baseline", results)
        self.assertIn("logistic_without_overdue_history", results)
        self.assertTrue(results["logistic_without_overdue_history"]["ablation"]["removed_features"])

    def test_holdout_scores_use_frozen_primary_model(self) -> None:
        holdout_path = REPORTS / "holdout_risk_scores.csv"
        if not holdout_path.exists():
            self.skipTest("公开提交包按体积策略省略订单级完整评分文件")
        with holdout_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            first = next(csv.DictReader(handle))
        self.assertEqual(first["primary_model"], self.freeze["primary_model"])
        self.assertIn(first["high_risk"], {"0", "1"})
        self.assertTrue(first["model_top_contributions"])

    def test_probability_calibration_outputs_exist(self) -> None:
        self.assertIn("probability_policy", self.metrics)
        self.assertTrue((REPORTS / "calibration_table.csv").exists())
        self.assertIn(
            "expected_calibration_error", self.freeze["test_metrics"]
        )
        self.assertIn("environment", self.freeze)
        self.assertIn("primary_model_sha256", self.freeze["provenance"])


if __name__ == "__main__":
    unittest.main()
