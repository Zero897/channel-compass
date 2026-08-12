from __future__ import annotations

import json

from common import FEISHU_DIR, PROCESSED_DIR, SYNTHETIC_DIR, read_csv, write_csv


RESULT = "完成客户沟通，安排库存调拨，建议复核额度"


def run_demo() -> None:
    events = read_csv(FEISHU_DIR / "风险事件.csv")
    target_event = next(row for row in events if row["事件编号"] == "R010")
    if target_event["风险等级"] != "红色":
        raise ValueError("C003 双风险事件未达到红色，停止生成演示结果")
    tasks = read_csv(FEISHU_DIR / "处置任务.csv")
    target_task = tasks[0]
    trace = [
        {"step": 1, "state": "风险出现", "evidence": target_event["关键指标"]},
        {"step": 2, "state": "规则解释", "evidence": target_event["规则解释"]},
        {"step": 3, "state": "生成任务", "task_status": "待处理"},
        {"step": 4, "state": "人工处理中", "task_status": "处理中"},
        {"step": 5, "state": "人工完成", "task_status": "已完成", "result": RESULT},
    ]
    target_task["审批状态"] = "已确认"
    target_task["执行状态"] = "已完成"
    target_task["人工处置方案"] = "平衡干预"
    target_task["执行结果"] = RESULT
    target_task["预警有效性"] = "有效"
    write_csv(FEISHU_DIR / "处置任务_本地验收结果.csv", list(target_task), [target_task])
    intervention = {
        "intervention_id": "IV001",
        "risk_event_id": "R010",
        "customer_id": "C003",
        "action_type": "平衡干预",
        "owner": "成员B",
        "status": "已完成",
        "created_at": "2026-08-03T10:00:00+08:00",
        "completed_at": "2026-08-03T10:30:00+08:00",
        "result": RESULT,
        "data_source": "synthetic",
    }
    write_csv(SYNTHETIC_DIR / "intervention.csv", list(intervention), [intervention])
    trace_path = PROCESSED_DIR / "c003_demo_trace.json"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_demo()
