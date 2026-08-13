from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_control_store import (  # noqa: E402
    FeishuRunControlStore,
    RUN_CONTROL_FIELD_TYPES,
)


class FakeClient:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []

    def list_tables(self):
        return [{"name": "运行控制", "table_id": "tbl-run"}]

    def list_fields(self, table_id):
        self.assert_table(table_id)
        return [
            {"field_name": name, "type": field_type}
            for name, field_type in RUN_CONTROL_FIELD_TYPES.items()
        ]

    def batch_create(self, table_id, records):
        self.assert_table(table_id)
        self.created.extend(records)
        return [{"record_id": "rec-run", **records[0]}]

    def batch_update(self, table_id, records):
        self.assert_table(table_id)
        self.updated.extend(records)
        return records

    @staticmethod
    def assert_table(table_id):
        if table_id != "tbl-run":
            raise AssertionError(table_id)


class RunControlStoreTest(unittest.TestCase):
    def test_create_and_update_run_control_record(self) -> None:
        client = FakeClient()
        store = FeishuRunControlStore(client)
        record_id = store.create({"运行编号": "RUN-1", "运行状态": "待运行"})
        store.update(record_id, {"运行状态": "成功"})
        self.assertEqual(record_id, "rec-run")
        self.assertEqual(client.updated[0]["fields"], {"运行状态": "成功"})


if __name__ == "__main__":
    unittest.main()
