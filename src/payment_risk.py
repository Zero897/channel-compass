from __future__ import annotations


def evaluate_payment_risk(
    overdue_amount_ratio: float,
    on_time_payment_rate: float,
    credit_utilization_ratio: float,
    recent_average_overdue_days: float,
    previous_average_overdue_days: float,
) -> tuple[str, list[str]]:
    evidence: list[str] = []
    if overdue_amount_ratio > 0.20:
        evidence.append(f"逾期金额占比{overdue_amount_ratio:.1%}>20%")
    if on_time_payment_rate < 0.70:
        evidence.append(f"按时回款率{on_time_payment_rate:.1%}<70%")
    if credit_utilization_ratio > 0.85:
        evidence.append(f"信用额度使用率{credit_utilization_ratio:.1%}>85%")
    if recent_average_overdue_days > previous_average_overdue_days + 0.1:
        evidence.append(
            "平均逾期天数由"
            f"{previous_average_overdue_days:.1f}天升至{recent_average_overdue_days:.1f}天"
        )
    if len(evidence) >= 3:
        return "红色", evidence
    if evidence:
        return "黄色", evidence
    return "绿色", evidence
