from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from score_frozen_payment_risk import score_frozen_model  # noqa: E402


class FrozenInferenceTest(unittest.TestCase):
    def test_frozen_model_reproduces_current_holdout_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "scores.csv"
            report = Path(temporary) / "report.json"
            result = score_frozen_model(
                PROJECT_ROOT / "data" / "processed" / "order_features.csv",
                PROJECT_ROOT / "models" / "primary_payment_risk.joblib",
                output,
                report,
            )
            expected = pd.read_csv(PROJECT_ROOT / "data" / "reports" / "holdout_risk_scores.csv")
            actual = pd.read_csv(output)
            self.assertEqual(result["model_type"], "logistic_regression")
            self.assertEqual(len(actual), len(expected))
            self.assertEqual(actual["order_id"].astype(str).tolist(), expected["order_id"].astype(str).tolist())
            self.assertLess(
                float((actual["risk_probability"] - expected["risk_probability"]).abs().max()),
                1e-12,
            )
            self.assertEqual(actual["high_risk"].tolist(), expected["high_risk"].tolist())


if __name__ == "__main__":
    unittest.main()
