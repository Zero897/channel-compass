from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_submission_package import FEISHU_TABLE_FILES, build_package  # noqa: E402


class SubmissionPackageTest(unittest.TestCase):
    def test_package_is_portable_and_excludes_large_private_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "submission.zip"
            result = build_package(package)
            self.assertEqual(result["status"], "pass")
            with zipfile.ZipFile(package) as archive:
                names = set(archive.namelist())
                manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
            self.assertIn("src/run_portable_demo.py", names)
            self.assertIn("requirements-lock.txt", names)
            self.assertIn("demo_output/model_metrics.json", names)
            self.assertFalse(any(name.startswith("data/company/") for name in names))
            self.assertNotIn("data/reports/holdout_risk_scores.csv", names)
            packaged_feishu = {
                Path(name).name
                for name in names
                if name.startswith("data/exports/feishu/")
            }
            self.assertEqual(packaged_feishu, FEISHU_TABLE_FILES)
            self.assertNotIn("data/exports/feishu/企业流程事件链.csv", names)
            self.assertNotIn("data/exports/feishu/企业处置任务_演示记录.csv", names)
            self.assertIn("demo_output/企业处置任务_演示记录.csv", names)
            self.assertIn("install", manifest["reproduction"])
            self.assertIn("tests", manifest["reproduction"])
            self.assertEqual(len(manifest["files"]) + 1, result["file_count"])


if __name__ == "__main__":
    unittest.main()
