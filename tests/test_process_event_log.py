from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_process_event_log import _task_sla  # noqa: E402


class ProcessEventLogTest(unittest.TestCase):
    def test_sla_accepts_timezone_aware_completion_and_date_only_deadline(self) -> None:
        task = {
            "截止日期": "2026-08-15",
            "完成时间": "2026-08-13T09:27:20.456000+08:00",
            "执行状态": "已完成",
        }

        self.assertEqual(_task_sla(task), "按期完成")


if __name__ == "__main__":
    unittest.main()
