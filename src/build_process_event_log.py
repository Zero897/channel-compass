from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

from project_paths import portable_path


ALLOWED_RESPONSIBILITY_ROLES = {
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


def _task_sla(task: dict[str, object]) -> str:
    due = pd.to_datetime(str(task.get("截止日期", "")), errors="coerce")
    completed = pd.to_datetime(str(task.get("完成时间", "")), errors="coerce")
    status = str(task.get("执行状态", ""))
    if status == "已完成" and pd.notna(completed) and pd.notna(due):
        return "按期完成" if completed.date() <= due.date() else "超时完成"
    if pd.notna(due) and due.date() < date.today():
        return "已超时"
    return "SLA内待处理"


def build_process_event_log(
    events_path: Path,
    tasks_path: Path,
    evidence_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, object]:
    events = pd.read_csv(events_path, dtype="string").fillna("")
    tasks = pd.read_csv(tasks_path, dtype="string").fillna("")
    evidence = pd.read_csv(evidence_path, dtype="string").fillna("")
    tasks_by_event = {row["风险事件编号"]: row for row in tasks.to_dict("records")}
    evidence_by_event = {
        event_id: group.to_dict("records")
        for event_id, group in evidence.groupby("风险事件编号")
    }
    rows: list[dict[str, object]] = []
    sequence = 0
    for event in events.to_dict("records"):
        event_id = event["事件编号"]
        customer_id = event["客户编号"]
        if event.get("模型版本", "") != "business_rule":
            for order in evidence_by_event.get(event_id, []):
                sequence += 1
                rows.append(
                    {
                        "流程序号": sequence,
                        "风险事件编号": event_id,
                        "客户编号": customer_id,
                        "关联订单号": order["销售订单号"],
                        "事件时间": order["出库日期"],
                        "流程阶段": "订单出库",
                        "事件说明": "开放订单进入后续回款风险观察",
                        "责任角色": "销售/商务",
                        "SLA状态": "不适用",
                        "数据性质": "原始业务时间",
                    }
                )
        sequence += 1
        rows.append(
            {
                "流程序号": sequence,
                "风险事件编号": event_id,
                "客户编号": customer_id,
                "关联订单号": "",
                "事件时间": event["数据快照"],
                "流程阶段": "应收快照观察",
                "事件说明": (
                    f"{event['风险类型']} / {event.get('动态到期阶段', '')}"
                    if event.get("动态到期阶段", "")
                    else event["风险类型"]
                ),
                "责任角色": "系统",
                "SLA状态": "不适用",
                "数据性质": "原始业务快照",
            }
        )
        sequence += 1
        rows.append(
            {
                "流程序号": sequence,
                "风险事件编号": event_id,
                "客户编号": customer_id,
                "关联订单号": "",
                "事件时间": event["触发时间"],
                "流程阶段": "风险预警",
                "事件说明": event["关键指标"],
                "责任角色": "渠智罗盘",
                "SLA状态": "不适用",
                "数据性质": "系统计算事件",
            }
        )
        task = tasks_by_event.get(event_id)
        if task:
            sla = _task_sla(task)
            sequence += 1
            rows.append(
                {
                    "流程序号": sequence,
                    "风险事件编号": event_id,
                    "客户编号": customer_id,
                    "关联订单号": "",
                    "事件时间": task.get("创建时间", event["触发时间"]),
                    "流程阶段": "处置任务",
                    "事件说明": f"{task['审批状态']} / {task['执行状态']}",
                    "责任角色": "任务负责人",
                    "SLA状态": sla,
                    "数据性质": "系统任务状态",
                }
            )
            if task.get("完成时间"):
                sequence += 1
                rows.append(
                    {
                        "流程序号": sequence,
                        "风险事件编号": event_id,
                        "客户编号": customer_id,
                        "关联订单号": "",
                        "事件时间": task["完成时间"],
                        "流程阶段": "结果回流",
                        "事件说明": task.get("执行结果", ""),
                        "责任角色": "任务负责人",
                        "SLA状态": sla,
                        "数据性质": "人工反馈",
                    }
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(rows)
    if not output.empty:
        invalid_roles = sorted(
            set(output["责任角色"].astype(str)) - ALLOWED_RESPONSIBILITY_ROLES
        )
        if invalid_roles:
            raise ValueError(f"责任角色包含飞书未配置的选项：{', '.join(invalid_roles)}")
        output["关联任务编号"] = output["风险事件编号"].map(
            lambda event_id: tasks_by_event.get(event_id, {}).get("任务编号", "")
        )
        output["对象类型"] = output["流程阶段"].map(
            {
                "订单出库": "销售订单",
                "应收快照观察": "客户应收",
                "风险预警": "风险事件",
                "处置任务": "处置任务",
                "结果回流": "处置任务",
            }
        )
        output["对象编号"] = output.apply(
            lambda row: (
                row["关联订单号"]
                if row["流程阶段"] == "订单出库"
                else tasks_by_event.get(row["风险事件编号"], {}).get("任务编号", "")
                if row["流程阶段"] in {"处置任务", "结果回流"}
                else row["客户编号"]
                if row["流程阶段"] == "应收快照观察"
                else row["风险事件编号"]
            ),
            axis=1,
        )
        output["事件类型"] = output["流程阶段"]
        output["时间戳"] = output["事件时间"]
        output["过程事件编号"] = output.apply(
            lambda row: "PE-"
            + hashlib.sha256(
                "|".join(
                    str(row[field])
                    for field in ("风险事件编号", "对象类型", "对象编号", "事件类型", "时间戳")
                ).encode("utf-8")
            ).hexdigest()[:12].upper(),
            axis=1,
        )
        output = output[
            [
                "过程事件编号",
                "对象类型",
                "对象编号",
                "事件类型",
                "时间戳",
                *[column for column in output.columns if column not in {"过程事件编号", "对象类型", "对象编号", "事件类型", "时间戳"}],
            ]
        ]
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    stage_counts = output["流程阶段"].value_counts().to_dict() if len(output) else {}
    sla_values = [_task_sla(row) for row in tasks.to_dict("records")]
    variants = (
        output.sort_values(["风险事件编号", "流程序号"])
        .groupby("风险事件编号")["流程阶段"]
        .apply(lambda values: " → ".join(values))
        .value_counts()
        .to_dict()
        if len(output)
        else {}
    )
    warning_times = {
        row["事件编号"]: pd.to_datetime(row["触发时间"], errors="coerce", utc=True)
        for row in events.to_dict("records")
    }
    task_delays = []
    for task in tasks.to_dict("records"):
        warning = warning_times.get(task["风险事件编号"])
        created = pd.to_datetime(task.get("创建时间", ""), errors="coerce", utc=True)
        if pd.notna(warning) and pd.notna(created):
            task_delays.append(max((created - warning).total_seconds() / 3600.0, 0.0))
    sla_violations = sum(value in {"已超时", "超时完成"} for value in sla_values)
    report = {
        "status": "pass",
        "event_rows": int(len(output)),
        "stage_counts": {str(key): int(value) for key, value in stage_counts.items()},
        "pending_approval_tasks": int((tasks["审批状态"] == "待审批").sum()),
        "overdue_sla_tasks": int(sla_violations),
        "sla_violation_rate": round(sla_violations / len(sla_values), 6) if sla_values else 0.0,
        "average_warning_to_task_hours": round(sum(task_delays) / len(task_delays), 3) if task_delays else None,
        "process_variants": {str(key): int(value) for key, value in variants.items()},
        "feedback_rows": int(stage_counts.get("结果回流", 0)),
        "boundary": "风险处置事件时间线，不冒充完整业务过程挖掘；仅展示可追溯对象事件、流程变体和当前SLA状态",
        "output": portable_path(output_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="生成对象化风险处置事件时间线")
    parser.add_argument("events", type=Path)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = build_process_event_log(
        args.events, args.tasks, args.evidence, args.output, args.report
    )
    print(
        f"风险处置时间线生成完成：{report['event_rows']}条事件，"
        f"{report['feedback_rows']}条结果回流。"
    )


if __name__ == "__main__":
    main()
