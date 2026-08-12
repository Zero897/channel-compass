from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from project_paths import portable_path


FEEDBACK_FIELDS = [
    "任务编号",
    "采用动作",
    "审批状态",
    "执行状态",
    "实际回款金额",
    "完成时间",
    "执行结果",
    "预警有效性",
    "备注",
]


def import_feedback(
    tasks_path: Path,
    feedback_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, object]:
    tasks = pd.read_csv(tasks_path, dtype="string").fillna("")
    feedback = pd.read_csv(feedback_path, dtype="string").fillna("")
    missing = [field for field in FEEDBACK_FIELDS if field not in feedback.columns]
    if missing:
        raise ValueError(f"反馈表缺少字段：{', '.join(missing)}")
    feedback = feedback[feedback["任务编号"].str.strip() != ""].copy()
    if feedback["任务编号"].duplicated().any():
        raise ValueError("反馈表存在重复任务编号")
    unknown = sorted(set(feedback["任务编号"]) - set(tasks["任务编号"]))
    if unknown:
        raise ValueError(f"反馈表包含未知任务编号：{', '.join(unknown[:5])}")
    allowed_validity = {"", "有效", "部分有效", "误报", "待确认"}
    invalid = sorted(set(feedback["预警有效性"]) - allowed_validity)
    if invalid:
        raise ValueError(f"预警有效性取值无效：{', '.join(invalid)}")

    merged = tasks.merge(feedback, on="任务编号", how="left", suffixes=("", "_反馈"))
    for field in FEEDBACK_FIELDS[1:]:
        merged[field] = merged[field].fillna("")
    update_map = {
        "采用动作": "人工处置方案",
        "审批状态_反馈": "审批状态",
        "执行状态_反馈": "执行状态",
        "完成时间_反馈": "完成时间",
        "实际回款金额_反馈": "实际回款金额",
        "执行结果_反馈": "执行结果",
        "预警有效性_反馈": "预警有效性",
        "备注": "反馈备注",
    }
    for source, target in update_map.items():
        if source not in merged:
            continue
        mask = merged[source].str.strip() != ""
        merged.loc[mask, target] = merged.loc[mask, source]
    drop_columns = [column for column in merged.columns if column.endswith("_反馈")]
    merged = merged.drop(columns=drop_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    completed = merged["执行状态"] == "已完成"
    valid = merged["预警有效性"] == "有效"
    report = {
        "status": "pass",
        "task_rows": int(len(tasks)),
        "feedback_rows": int(len(feedback)),
        "completed_tasks": int(completed.sum()),
        "completion_rate": round(float(completed.mean()), 6) if len(merged) else 0.0,
        "confirmed_effective_alerts": int(valid.sum()),
        "boundary": "反馈由人工填写或从飞书导出后导入；当前脚本不声称已接通飞书API",
        "output": portable_path(output_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="校验并合并飞书处置任务反馈")
    parser.add_argument("tasks", type=Path)
    parser.add_argument("feedback", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = import_feedback(args.tasks, args.feedback, args.output, args.report)
    print(
        f"任务反馈导入完成：{report['feedback_rows']}条反馈，"
        f"{report['completed_tasks']}条任务已完成。"
    )


if __name__ == "__main__":
    main()
