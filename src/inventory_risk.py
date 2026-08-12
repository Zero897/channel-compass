from __future__ import annotations


def evaluate_inventory_risk(
    coverage_weeks: float,
    average_age_days: float,
    sales_growth_ratio: float,
) -> tuple[str, list[str]]:
    evidence: list[str] = []
    if coverage_weeks > 8:
        evidence.append(f"库存覆盖{coverage_weeks:.1f}周>8周")
    if average_age_days > 90:
        evidence.append(f"平均库龄{average_age_days:.1f}天>90天")
    if sales_growth_ratio < -0.20:
        evidence.append(f"近4周销量下降{abs(sales_growth_ratio):.1%}>20%")
    if len(evidence) >= 3:
        return "红色", evidence
    if len(evidence) == 2:
        return "黄色", evidence
    return "绿色", evidence
