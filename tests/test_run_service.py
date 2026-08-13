from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_service import RunBusyError, RunManager  # noqa: E402


class FakeRunStore:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.updated: list[tuple[str, dict[str, object]]] = []

    def create(self, fields: dict[str, object]) -> str:
        self.created.append(fields)
        return "rec-control"

    def update(self, record_id: str, fields: dict[str, object]) -> None:
        self.updated.append((record_id, fields))


class RunServiceTest(unittest.TestCase):
    def test_submit_returns_before_background_job_finishes_and_writes_status(self) -> None:
        release = threading.Event()
        started = threading.Event()
        store = FakeRunStore()

        def runner(progress):
            progress("冻结模型推理")
            started.set()
            release.wait(5)
            return {"updated": 7, "created": 2, "report_path": "report.json"}

        manager = RunManager(runner, store)
        job = manager.submit(requested_by="测试人员")
        self.assertTrue(started.wait(2))
        self.assertEqual(manager.get(job.run_id).status, "运行中")
        release.set()
        finished = manager.wait(job.run_id, timeout=5)
        self.assertEqual(finished.status, "成功")
        self.assertEqual(finished.updated, 7)
        self.assertEqual(finished.created, 2)
        self.assertTrue(any(fields.get("运行状态") == "成功" for _, fields in store.updated))

    def test_only_one_run_can_execute_at_a_time(self) -> None:
        release = threading.Event()
        started = threading.Event()
        store = FakeRunStore()

        def runner(progress):
            started.set()
            release.wait(5)
            return {"updated": 0, "created": 0, "report_path": ""}

        manager = RunManager(runner, store)
        first = manager.submit(requested_by="A")
        self.assertTrue(started.wait(2))
        with self.assertRaises(RunBusyError):
            manager.submit(requested_by="B")
        release.set()
        manager.wait(first.run_id, timeout=5)


if __name__ == "__main__":
    unittest.main()
