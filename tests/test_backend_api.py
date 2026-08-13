from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backend_api import create_app  # noqa: E402


class FakeJob:
    run_id = "RUN-TEST"

    @staticmethod
    def to_dict():
        return {"run_id": "RUN-TEST", "status": "待运行"}


class FakeManager:
    def __init__(self) -> None:
        self.submitted = []

    def submit(self, *, requested_by, control_record_id=""):
        self.submitted.append((requested_by, control_record_id))
        return FakeJob()

    def get(self, run_id):
        if run_id != "RUN-TEST":
            from run_service import RunNotFoundError

            raise RunNotFoundError(run_id)
        return FakeJob()

    def shutdown(self):
        return None


class BackendAPITest(unittest.TestCase):
    def test_run_endpoint_requires_bearer_token_and_returns_202(self) -> None:
        manager = FakeManager()
        with TestClient(
            create_app(manager, trigger_token="secret", write_enabled=True)
        ) as client:
            unauthorized = client.post("/api/runs", json={"requested_by": "A"})
            accepted = client.post(
                "/api/runs",
                headers={"Authorization": "Bearer secret"},
                json={"requested_by": "A", "control_record_id": "rec-1"},
            )
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(accepted.status_code, 202)
        self.assertEqual(accepted.json()["run_id"], "RUN-TEST")
        self.assertEqual(manager.submitted, [("A", "rec-1")])

    def test_write_safety_switch_blocks_run(self) -> None:
        manager = FakeManager()
        with TestClient(
            create_app(manager, trigger_token="secret", write_enabled=False)
        ) as client:
            response = client.post(
                "/api/runs",
                headers={"Authorization": "Bearer secret"},
                json={},
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(manager.submitted, [])

    def test_run_endpoint_accepts_feishu_double_encoded_json(self) -> None:
        manager = FakeManager()
        body = json.dumps(
            {"requested_by": "飞书前端", "control_record_id": ""},
            ensure_ascii=False,
        )
        with TestClient(
            create_app(manager, trigger_token="secret", write_enabled=True)
        ) as client:
            response = client.post(
                "/api/runs",
                headers={
                    "Authorization": "Bearer secret",
                    "Content-Type": "application/json",
                },
                json=body,
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(manager.submitted, [("飞书前端", "")])


if __name__ == "__main__":
    unittest.main()
