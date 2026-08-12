from __future__ import annotations

from common import clamp


WEIGHTS = {
    "revenue_quality_score": 0.20,
    "inventory_health_score": 0.25,
    "payment_ability_score": 0.25,
    "credit_exposure_score": 0.20,
    "cooperation_stability_score": 0.10,
}


def calculate_dimension_scores(features: dict[str, float]) -> dict[str, float]:
    sales_growth = features["sales_growth_ratio"]
    revenue_score = clamp(92 + sales_growth * 120)
    inventory_score = clamp(
        100
        - max(0.0, features["inventory_coverage_weeks"] - 5.0) * 8.0
        - max(0.0, features["average_inventory_age_days"] - 60.0) * 0.6
    )
    payment_score = clamp(
        100
        - features["overdue_amount_ratio"] * 70
        - (1.0 - features["on_time_payment_rate"]) * 40
        - min(features["recent_average_overdue_days"] * 1.5, 20)
    )
    utilization = features["credit_utilization_ratio"]
    credit_score = clamp(100 - max(0.0, utilization - 0.60) * 180)
    stability_score = clamp(60 + features["cooperation_months"] * 0.55)
    return {
        "revenue_quality_score": round(revenue_score, 1),
        "inventory_health_score": round(inventory_score, 1),
        "payment_ability_score": round(payment_score, 1),
        "credit_exposure_score": round(credit_score, 1),
        "cooperation_stability_score": round(stability_score, 1),
    }


def calculate_health_score(dimension_scores: dict[str, float]) -> tuple[float, str]:
    score = sum(dimension_scores[name] * weight for name, weight in WEIGHTS.items())
    score = round(score, 1)
    if score >= 80:
        return score, "绿色"
    if score >= 60:
        return score, "黄色"
    return score, "红色"
