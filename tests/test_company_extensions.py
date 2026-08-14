from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "data" / "reports"
FEISHU = PROJECT_ROOT / "data" / "exports" / "feishu"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from import_task_feedback import import_feedback  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class CompanyExtensionsTest(unittest.TestCase):
    def test_inventory_keeps_sku_warehouse_scope(self) -> None:
        audit = json.loads(
            (REPORTS / "inventory_health_audit.json").read_text(encoding="utf-8")
        )
        rows = _read_csv(FEISHU / "企业库存风险.csv")
        self.assertEqual(audit["status"], "pass")
        self.assertTrue(rows)
        self.assertLessEqual(len(rows), 500)
        self.assertNotIn("客户编号", rows[0])
        self.assertTrue(all("不归因到客户" in row["对象口径"] for row in rows))
        self.assertTrue((FEISHU / "企业库存风险汇总.csv").exists())
        self.assertEqual(len({row["库存对象编号"] for row in rows}), len(rows))
        summary = _read_csv(FEISHU / "企业库存风险汇总.csv")
        self.assertEqual(len({row["库存汇总编号"] for row in summary}), len(summary))
        self.assertTrue(audit["detail_export_truncated"])
        self.assertIn("demand_baseline", audit)
        self.assertTrue(all("未来4周需求基线" in row for row in rows))
        self.assertTrue(all("未来8周需求基线" in row for row in rows))

    def test_dynamic_due_rules_follow_enterprise_feedback(self) -> None:
        audit = json.loads(
            (REPORTS / "prototype_alignment_audit.json").read_text(encoding="utf-8")
        )
        rows = _read_csv(FEISHU / "企业动态回款监控.csv")
        allowed = {
            "数据待刷新",
            "未临期",
            "到期前5天",
            "超期1-30天",
            "超期31-60天",
            "超期61-120天",
            "超期120天以上",
        }
        self.assertEqual(audit["status"], "pass")
        self.assertTrue(rows)
        self.assertTrue(all(row["动态到期阶段"] in allowed for row in rows))
        self.assertTrue(all(row["动态建议动作"] for row in rows))
        self.assertTrue(all("人工审批" in row["规则口径"] for row in rows))
        self.assertTrue(all(row["运行模式"] in {"historical_replay", "business_current"} for row in rows))
        self.assertTrue(all(row["计算基准日"] for row in rows))
        self.assertTrue(all(row["数据更新时间"] for row in rows))
        self.assertTrue(all(int(row["数据新鲜度天数"]) >= 0 for row in rows))

    def test_five_dimension_health_is_explainable(self) -> None:
        rows = _read_csv(FEISHU / "企业渠道客户.csv")
        self.assertEqual(next(iter(rows[0])), "客户主键")
        dimensions = ["营收质量分", "库存周转暴露分", "付款行为分", "信用暴露分", "合作稳定性分"]
        for row in rows:
            self.assertTrue(all(0 <= float(row[name]) <= 100 for name in dimensions))
            self.assertTrue(0 <= float(row["综合健康度"]) <= 100)
            self.assertIn("采购暴露", row["健康度口径"])

    def test_customer_product_exposure_does_not_claim_customer_inventory(self) -> None:
        rows = _read_csv(FEISHU / "企业客户商品风险暴露.csv")
        self.assertTrue(rows)
        self.assertTrue(all("不代表客户持有库存" in row["对象口径"] for row in rows))
        self.assertEqual(len({row["采购暴露编号"] for row in rows}), len(rows))
        self.assertTrue(all(row["公司端库存金额（快照）"] for row in rows))
        self.assertTrue(all(row["库存快照日期"] for row in rows))
        inventory_rows = _read_csv(FEISHU / "企业库存风险.csv")
        expected_snapshot = max(row["快照日期"] for row in inventory_rows)
        self.assertTrue(all(row["库存快照日期"] == expected_snapshot for row in rows))

    def test_cross_table_references_are_complete(self) -> None:
        audit = json.loads(
            (REPORTS / "prototype_alignment_audit.json").read_text(encoding="utf-8")
        )["cross_reference_audit"]
        self.assertEqual(audit["status"], "pass")
        self.assertGreater(audit["customers_added_to_master"], 0)
        self.assertGreater(audit["pre_fix_exposure_orphan_rows"], 0)
        self.assertEqual(audit["exposure_orphan_customer_rows"], 0)
        self.assertEqual(audit["risk_event_orphan_customer_rows"], 0)
        self.assertEqual(audit["task_orphan_event_rows"], 0)

    def test_task_card_contains_decision_context(self) -> None:
        rows = _read_csv(FEISHU / "企业处置任务.csv")
        required = {"风险类型", "风险等级", "影响金额", "关键证据", "置信说明", "动态到期阶段", "审批建议", "建议负责人角色", "SLA状态"}
        self.assertTrue(rows)
        self.assertTrue(required.issubset(rows[0]))

    def test_three_scenarios_are_transparent_not_causal(self) -> None:
        rows = _read_csv(FEISHU / "企业处置情景.csv")
        by_customer: dict[str, set[str]] = {}
        for row in rows:
            by_customer.setdefault(row["客户编号"], set()).add(row["方案名称"])
            self.assertIn("不代表因果效果", row["测算性质"])
        self.assertTrue(by_customer)
        self.assertTrue(all(len(names) == 3 for names in by_customer.values()))

    def test_process_log_does_not_invent_feedback(self) -> None:
        report = json.loads(
            (REPORTS / "process_event_log_audit.json").read_text(encoding="utf-8")
        )
        rows = _read_csv(FEISHU / "企业风险处置时间线.csv")
        tasks = _read_csv(FEISHU / "企业处置任务.csv")
        completed_tasks = {
            row["任务编号"] for row in tasks if row.get("完成时间", "").strip()
        }
        feedback_rows = [row for row in rows if row["流程阶段"] == "结果回流"]
        self.assertGreater(len(rows), 0)
        self.assertEqual(report["feedback_rows"], len(completed_tasks))
        self.assertEqual(len(feedback_rows), len(completed_tasks))
        self.assertEqual(
            {row["关联任务编号"] for row in feedback_rows}, completed_tasks
        )

    def test_process_log_responsibility_roles_match_feishu_options(self) -> None:
        rows = _read_csv(FEISHU / "企业风险处置时间线.csv")
        allowed_roles = {
            "系统",
            "渠智罗盘",
            "销售/商务",
            "客户经理",
            "财务",
            "信用管理",
            "法务",
            "审批人",
            "任务负责人",
        }
        self.assertTrue({row["责任角色"] for row in rows} <= allowed_roles)
        task_rows = [
            row
            for row in rows
            if row["流程阶段"] in {"处置任务", "结果回流"}
        ]
        self.assertGreater(len(task_rows), 0)
        self.assertEqual(
            {row["责任角色"] for row in task_rows}, {"任务负责人"}
        )
        self.assertTrue(all(row["过程事件编号"].startswith("PE-") for row in rows))
        self.assertTrue(all(row["对象类型"] for row in rows))
        self.assertTrue(all("关联任务编号" in row for row in rows))

    def test_early_warning_claim_is_evidence_gated(self) -> None:
        report = json.loads(
            (REPORTS / "early_warning_evidence_audit.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["strict_customer_early_warning_cases"], 0)
        self.assertIn("未发现", report["claim_policy"])
        self.assertTrue((PROJECT_ROOT / "demo_output" / "企业处置任务_演示记录.csv").exists())

    def test_feedback_import_validates_and_merges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = root / "tasks.csv"
            feedback = root / "feedback.csv"
            output = root / "merged.csv"
            report = root / "report.json"
            tasks.write_text(
                "任务编号,风险事件编号,人工处置方案,审批状态,执行状态,执行结果,预警有效性,负责人,截止日期\n"
                "T001,R001,待填写,待审批,待处理,,待确认,成员A,2026-08-10\n",
                encoding="utf-8-sig",
            )
            feedback.write_text(
                "任务编号,采用动作,审批状态,执行状态,实际回款金额,完成时间,执行结果,预警有效性,备注\n"
                "T001,电话核查,已批准,已完成,1000,2026-08-09,已确认回款计划,有效,测试\n",
                encoding="utf-8-sig",
            )
            result = import_feedback(tasks, feedback, output, report)
            merged = _read_csv(output)[0]
            self.assertEqual(result["completed_tasks"], 1)
            self.assertEqual(merged["人工处置方案"], "电话核查")
            self.assertEqual(merged["预警有效性"], "有效")

    def test_feedback_import_accepts_partially_effective(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = root / "tasks.csv"
            feedback = root / "feedback.csv"
            output = root / "merged.csv"
            report = root / "report.json"
            tasks.write_text(
                "任务编号,人工处置方案,审批状态,执行状态,完成时间,实际回款金额,执行结果,预警有效性,反馈备注\n"
                "T001,待填写,待审批,待处理,,,,待确认,\n",
                encoding="utf-8-sig",
            )
            feedback.write_text(
                "任务编号,采用动作,审批状态,执行状态,实际回款金额,完成时间,执行结果,预警有效性,备注\n"
                "T001,电话核查,已批准,已完成,1000,2026-08-09,部分证据成立,部分有效,测试\n",
                encoding="utf-8-sig",
            )
            import_feedback(tasks, feedback, output, report)
            self.assertEqual(_read_csv(output)[0]["预警有效性"], "部分有效")


if __name__ == "__main__":
    unittest.main()
