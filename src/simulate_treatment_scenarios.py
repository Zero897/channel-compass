from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from project_paths import portable_path


PREDICTIVE_SCENARIOS = (
    ("不处理｜继续原条件履约", 1.0, 500.0, "低", "保留全部业务机会，但新增应收暴露最高"),
    ("平衡干预｜缩减50%赊销额度", 0.5, 1500.0, "中", "保留部分业务机会，并降低新增应收暴露"),
    ("保守干预｜暂停新增赊销履约", 0.0, 3000.0, "高", "不增加赊销暴露，需人工评估客户关系与现金交易替代方案"),
)
OVERDUE_SCENARIOS = (
    ("不处理｜正常催收并保留新增赊销", 1.0, 800.0, "低", "保持合作，但资金占用与风险扩散暴露较高"),
    ("平衡干预｜协商分期并缩减50%新增赊销", 0.5, 2000.0, "中", "以分期和限额平衡回款与客户关系"),
    ("保守干预｜暂停新增赊销并专项催收", 0.0, 4500.0, "高", "停止新增赊销暴露，需审批并评估收入与客户影响"),
)
SENSITIVITY_FACTORS = {"低": 0.75, "基准": 1.0, "高": 1.25}
ANNUAL_CAPITAL_COST_RATE = 0.06
CAPITAL_OCCUPATION_DAYS = 30


def simulate_scenarios(
    customer_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, object]:
    customers = pd.read_csv(customer_path, low_memory=False)
    candidates = customers[customers["风险等级"] != "绿色"].copy()
    rows: list[dict[str, object]] = []
    for customer in candidates.to_dict("records"):
        order_count = max(int(customer["预测高风险订单数"]), 1)
        predictive_amount = float(customer["预测高风险应收金额"])
        current_ar = float(customer["当前应收余额"])
        planned_order = max(predictive_amount / order_count, current_ar * 0.05, 0.0)
        probability = float(customer["最高模型概率"])
        if probability <= 0:
            probability = min(max(float(customer["逾期占比"]), 0.05), 1.0)
        current_weighted_exposure = float(customer["风险加权应收暴露"])
        existing_overdue = float(customer["逾期应收金额"])
        scenarios = OVERDUE_SCENARIOS if existing_overdue > 0 else PREDICTIVE_SCENARIOS
        continue_exposure = current_weighted_exposure + planned_order * probability
        for index, (name, approval_ratio, execution_cost, impact, boundary) in enumerate(scenarios, start=1):
            approved_amount = planned_order * approval_ratio
            weighted_exposure = current_weighted_exposure + approved_amount * probability
            sensitivity = {
                level: current_weighted_exposure
                + approved_amount * min(probability * factor, 1.0)
                for level, factor in SENSITIVITY_FACTORS.items()
            }
            capital_cost = weighted_exposure * ANNUAL_CAPITAL_COST_RATE * CAPITAL_OCCUPATION_DAYS / 365
            rows.append(
                {
                    "方案编号": f"{customer['客户编号']}-S{index}",
                    "客户编号": customer["客户编号"],
                    "客户名称": customer["客户名称"],
                    "方案名称": name,
                    "测算订单金额": round(planned_order, 2),
                    "赊销批准比例": approval_ratio,
                    "保留业务机会金额": round(approved_amount, 2),
                    "测算后风险加权应收暴露": round(weighted_exposure, 2),
                    "低档风险暴露": round(sensitivity["低"], 2),
                    "基准风险暴露": round(sensitivity["基准"], 2),
                    "高档风险暴露": round(sensitivity["高"], 2),
                    "相对继续履约降低暴露": round(continue_exposure - weighted_exposure, 2),
                    "30天资金占用成本": round(capital_cost, 2),
                    "预计执行成本": round(execution_cost, 2),
                    "潜在收入机会减少": round(planned_order - approved_amount, 2),
                    "客户影响": impact,
                    "适用边界": boundary,
                    "人工决策": "待选择",
                    "测算性质": "参数化What-if，不代表因果效果或坏账预测",
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    report = {
        "status": "pass",
        "customers": int(len(candidates)),
        "customers_with_existing_overdue": int((candidates["逾期应收金额"] > 0).sum()),
        "scenario_rows": len(rows),
        "assumptions": {
            "planned_order": "每个客户当前高风险开放应收金额除以高风险订单数，作为一笔代表性新增订单",
            "probability": "预测客户使用最高模型输出；仅存量逾期客户以逾期占比作为透明敏感度基线",
            "exposure": "当前风险加权应收暴露加上新增批准金额乘校准概率",
            "sensitivity": "低/基准/高档分别使用0.75/1.00/1.25倍风险系数并截断到1",
            "capital_cost": "按年化6%和30天资金占用期估算，不代表财务确认值",
        },
        "boundary": "仅供人工比较，不预测干预的因果效果，不自动决定授信或停供",
        "output": portable_path(output_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="生成三种处置策略的透明What-if测算")
    parser.add_argument("customers", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = simulate_scenarios(args.customers, args.output, args.report)
    print(f"三演测算完成：{report['customers']}个客户，{report['scenario_rows']}条方案。")


if __name__ == "__main__":
    main()
