from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_portable_demo import run_demo  # noqa: E402


class PortableDemoTest(unittest.TestCase):
    def test_demo_runs_from_data_to_model_and_page_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "demo"
            result = run_demo(output)
            self.assertEqual(result["status"], "pass")
            self.assertGreater(result["test_pr_auc"], 0)
            self.assertGreater(result["risk_events"], 0)
            for filename in result["outputs"]:
                self.assertTrue((output / filename).exists(), filename)
            saved = json.loads((output / "model_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["data_source"], "portable_synthetic_demo")


if __name__ == "__main__":
    unittest.main()
