from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pull_feishu_feedback import OUTPUT_FIELDS, pull_feedback  # noqa: E402
from sync_feishu import (  # noqa: E402
    build_upsert_plan,
    convert_for_feishu,
    load_sync_config,
    run_sync,
)


class FakeFeishuClient:
    def __init__(
        self,
        tables: list[dict[str, Any]],
        fields: dict[str, list[dict[str, Any]]],
        records: dict[str, list[dict[str, Any]]],
    ) -> None:
        self.tables = tables
        self.fields = fields
        self.records = records
        self.created: list[tuple[str, list[dict[str, Any]]]] = []
        self.updated: list[tuple[str, list[dict[str, Any]]]] = []
        self.masked_app_token = "app***test"

    def list_tables(self) -> list[dict[str, Any]]:
        return self.tables

    def list_fields(self, table_id: str) -> list[dict[str, Any]]:
        return self.fields[table_id]

    def search_records(
        self, table_id: str, *, field_names: Any = None
    ) -> list[dict[str, Any]]:
        del field_names
        return self.records[table_id]

    def batch_create(
        self, table_id: str, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self.created.append((table_id, records))
        return [
            {"record_id": f"new-{index}", **record}
            for index, record in enumerate(records)
        ]

    def batch_update(
        self, table_id: str, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self.updated.append((table_id, records))
        return records


class FeishuSyncTest(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        (source / "risk.csv").write_text(
            "事件编号,客户名称\nR001,新名称\nR002,新增客户\n",
            encoding="utf-8-sig",
        )
        config = {
            "source_dir": str(source),
            "reports_dir": str(root / "reports"),
            "batch_size": 100,
            "request_timeout_seconds": 30,
            "tables": [
                {
                    "local_file": "risk.csv",
                    "remote_table": "风险事件",
                    "stable_key": "事件编号",
                    "update_fields": ["客户名称"],
                    "create_defaults": {"处理状态": "待处理"},
                    "protected_fields": ["处理状态", "AI风险摘要"],
                }
            ],
        }
        path = root / "sync.json"
        path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        return path

    @staticmethod
    def _risk_client() -> FakeFeishuClient:
        fields = [
            {"field_name": "事件编号", "type": 1},
            {"field_name": "客户名称", "type": 1},
            {"field_name": "处理状态", "type": 3},
            {"field_name": "AI风险摘要", "type": 1},
        ]
        return FakeFeishuClient(
            [{"name": "风险事件", "table_id": "tbl-risk"}],
            {"tbl-risk": fields},
            {
                "tbl-risk": [
                    {
                        "record_id": "rec-1",
                        "fields": {
                            "事件编号": [{"text": "R001", "type": "text"}],
                            "客户名称": [{"text": "旧名称", "type": "text"}],
                            "处理状态": "处理中",
                            "AI风险摘要": "人工保留内容",
                        },
                    }
                ]
            },
        )

    def test_production_config_uses_explicit_non_overlapping_whitelists(self) -> None:
        config = load_sync_config(PROJECT_ROOT / "config" / "feishu_sync.json")
        self.assertEqual(len(config["tables"]), 11)
        self.assertEqual(
            {table["remote_table"] for table in config["tables"]},
            {
                "渠道客户",
                "风险事件",
                "处置任务",
                "订单证据",
                "处置情景",
                "风险处置时间线",
                "动态回款监控",
                "客户商品风险暴露",
                "库存风险",
                "库存风险汇总",
                "模型指标",
            },
        )
        for table in config["tables"]:
            self.assertTrue(table["update_fields"])
            self.assertFalse(
                set(table["update_fields"]) & set(table.get("protected_fields", []))
            )
        scenario = next(
            table for table in config["tables"] if table["remote_table"] == "处置情景"
        )
        self.assertIn("人工决策", scenario["protected_fields"])
        self.assertNotIn("人工决策", scenario["update_fields"])

    def test_build_plan_updates_only_whitelisted_fields(self) -> None:
        table_config = {
            "local_file": "risk.csv",
            "remote_table": "风险事件",
            "stable_key": "事件编号",
            "update_fields": ["客户名称"],
            "create_defaults": {"处理状态": "待处理"},
            "protected_fields": ["处理状态", "AI风险摘要"],
        }
        local = [
            {"事件编号": "R001", "客户名称": "新名称"},
            {"事件编号": "R002", "客户名称": "新增客户"},
        ]
        client = self._risk_client()
        plan = build_upsert_plan(
            table_config,
            local,
            client.records["tbl-risk"],
            client.fields["tbl-risk"],
        )
        self.assertEqual(len(plan["creates"]), 1)
        self.assertEqual(len(plan["updates"]), 1)
        update = plan["updates"][0]
        self.assertEqual(update["fields"], {"客户名称": "新名称"})
        self.assertNotIn("处理状态", update["fields"])
        self.assertNotIn("AI风险摘要", update["fields"])
        self.assertEqual(
            plan["creates"][0]["fields"]["处理状态"], "待处理"
        )

    def test_build_plan_maps_local_field_to_remote_field(self) -> None:
        table_config = {
            "local_file": "risk.csv",
            "remote_table": "风险事件",
            "stable_key": "事件编号",
            "update_fields": ["动态建议动作"],
            "field_mappings": {"动态建议动作": "分级处置建议"},
            "protected_fields": [],
        }
        fields = [
            {"field_name": "事件编号", "type": 1},
            {"field_name": "分级处置建议", "type": 1},
        ]
        plan = build_upsert_plan(
            table_config,
            [{"事件编号": "R001", "动态建议动作": "电话提醒"}],
            [
                {
                    "record_id": "rec-1",
                    "fields": {"事件编号": "R001", "分级处置建议": "邮件提醒"},
                }
            ],
            fields,
        )
        self.assertEqual(
            plan["updates"][0]["fields"], {"分级处置建议": "电话提醒"}
        )

    def test_build_plan_reports_and_ignores_remote_row_without_stable_key(self) -> None:
        table_config = {
            "local_file": "risk.csv",
            "remote_table": "风险事件",
            "stable_key": "事件编号",
            "update_fields": ["客户名称"],
            "protected_fields": [],
        }
        fields = [
            {"field_name": "事件编号", "type": 1},
            {"field_name": "客户名称", "type": 1},
        ]
        plan = build_upsert_plan(
            table_config,
            [{"事件编号": "R001", "客户名称": "客户A"}],
            [
                {"record_id": "rec-empty", "fields": {"客户名称": "旧记录"}},
                {
                    "record_id": "rec-1",
                    "fields": {"事件编号": "R001", "客户名称": "客户A"},
                },
            ],
            fields,
        )
        self.assertEqual(plan["remote_missing_stable_key"], 1)
        self.assertEqual(plan["skipped"], 1)
        self.assertEqual(plan["creates"], [])
        self.assertEqual(plan["updates"], [])

    def test_create_only_field_is_written_only_for_new_record(self) -> None:
        table_config = {
            "local_file": "risk.csv",
            "remote_table": "风险事件",
            "stable_key": "事件编号",
            "update_fields": ["客户名称"],
            "create_only_fields": ["触发时间"],
            "protected_fields": [],
        }
        fields = [
            {"field_name": "事件编号", "type": 1},
            {"field_name": "客户名称", "type": 1},
            {"field_name": "触发时间", "type": 5},
        ]
        plan = build_upsert_plan(
            table_config,
            [
                {"事件编号": "R001", "客户名称": "客户A", "触发时间": "2026-08-13"},
                {"事件编号": "R002", "客户名称": "客户B", "触发时间": "2026-08-13"},
            ],
            [
                {
                    "record_id": "rec-1",
                    "fields": {
                        "事件编号": "R001",
                        "客户名称": "客户A",
                        "触发时间": 1786464000000,
                    },
                }
            ],
            fields,
        )
        self.assertEqual(plan["updates"], [])
        self.assertIn("触发时间", plan["creates"][0]["fields"])

    def test_dry_run_never_calls_write_methods(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._write_config(root)
            client = self._risk_client()
            report = run_sync(config_path, apply=False, client=client)
            self.assertEqual(report["mode"], "dry-run")
            self.assertEqual(report["tables"]["风险事件"]["planned_create"], 1)
            self.assertEqual(report["tables"]["风险事件"]["planned_update"], 1)
            self.assertEqual(client.created, [])
            self.assertEqual(client.updated, [])

    def test_apply_uses_create_and_update_without_protected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._write_config(root)
            client = self._risk_client()
            report = run_sync(config_path, apply=True, client=client)
            self.assertEqual(report["tables"]["风险事件"]["created"], 1)
            self.assertEqual(report["tables"]["风险事件"]["updated"], 1)
            update_fields = client.updated[0][1][0]["fields"]
            self.assertEqual(update_fields, {"客户名称": "新名称"})

    def test_invalid_protected_update_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "source_dir": str(root),
                "reports_dir": str(root),
                "batch_size": 100,
                "tables": [
                    {
                        "local_file": "a.csv",
                        "remote_table": "A",
                        "stable_key": "ID",
                        "update_fields": ["人工字段"],
                        "protected_fields": ["人工字段"],
                    }
                ],
            }
            path = root / "bad.json"
            path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "可更新字段与保护字段冲突"):
                load_sync_config(path)

    def test_date_conversion_uses_beijing_midnight(self) -> None:
        timestamp = convert_for_feishu(
            "2026-07-31", {"field_name": "快照日期", "type": 5}
        )
        self.assertEqual(timestamp, 1785427200000)

    def test_pull_feedback_writes_existing_import_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "source_dir": str(root),
                "reports_dir": str(root),
                "batch_size": 100,
                "tables": [
                    {
                        "local_file": "企业处置任务.csv",
                        "remote_table": "处置任务",
                        "stable_key": "任务编号",
                        "update_fields": ["AI建议动作"],
                        "protected_fields": ["人工处置方案"],
                    }
                ],
            }
            config_path = root / "sync.json"
            config_path.write_text(
                json.dumps(config, ensure_ascii=False), encoding="utf-8"
            )
            field_names = list(FEEDBACK_MAPPING_FOR_TEST)
            fields = [
                {
                    "field_name": name,
                    "type": 2 if name == "实际回款金额" else 5 if name == "完成时间" else 1,
                }
                for name in field_names
            ]
            client = FakeFeishuClient(
                [{"name": "处置任务", "table_id": "tbl-task"}],
                {"tbl-task": fields},
                {
                    "tbl-task": [
                        {
                            "record_id": "rec-task",
                            "fields": {
                                "任务编号": "T001",
                                "人工处置方案": "电话核查",
                                "审批状态": "已批准",
                                "执行状态": "已完成",
                                "实际回款金额": 1000.0,
                                "完成时间": 1785427200000,
                                "执行结果": "已确认计划",
                                "预警有效性": "有效",
                                "反馈备注": "人工确认",
                            },
                        }
                    ]
                },
            )
            output = root / "feedback.csv"
            report_path = root / "feedback.json"
            report = pull_feedback(
                config_path, output, report_path, client=client
            )
            with output.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0]), OUTPUT_FIELDS)
            self.assertEqual(rows[0]["任务编号"], "T001")
            self.assertEqual(rows[0]["实际回款金额"], "1000")
            self.assertEqual(rows[0]["完成时间"], "2026-07-31")
            self.assertEqual(report["feedback_rows"], 1)


FEEDBACK_MAPPING_FOR_TEST = [
    "任务编号",
    "人工处置方案",
    "审批状态",
    "执行状态",
    "实际回款金额",
    "完成时间",
    "执行结果",
    "预警有效性",
    "反馈备注",
]


if __name__ == "__main__":
    unittest.main()
