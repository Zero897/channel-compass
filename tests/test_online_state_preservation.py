from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aggregate_company_customer_risk import (  # noqa: E402
    _preserve_event_state,
    _preserve_task_state,
)


class OnlineStatePreservationTest(unittest.TestCase):
    def test_existing_event_and_task_dates_are_not_reset_by_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_path = root / "events.csv"
            task_path = root / "tasks.csv"
            pd.DataFrame(
                [{"事件编号": "E1", "触发时间": "2026-08-12"}]
            ).to_csv(event_path, index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {
                        "任务编号": "T1",
                        "创建时间": "2026-08-12",
                        "截止日期": "2026-08-14",
                        "审批状态": "待审批",
                    }
                ]
            ).to_csv(task_path, index=False, encoding="utf-8-sig")

            events = _preserve_event_state(
                pd.DataFrame([{"事件编号": "E1", "触发时间": "2026-08-13"}]),
                event_path,
            )
            tasks = _preserve_task_state(
                pd.DataFrame(
                    [
                        {
                            "任务编号": "T1",
                            "创建时间": "2026-08-13",
                            "截止日期": "2026-08-15",
                            "审批状态": "待审批",
                        }
                    ]
                ),
                task_path,
            )
            self.assertEqual(events.loc[0, "触发时间"], "2026-08-12")
            self.assertEqual(tasks.loc[0, "创建时间"], "2026-08-12")
            self.assertEqual(tasks.loc[0, "截止日期"], "2026-08-14")


if __name__ == "__main__":
    unittest.main()
