from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

from common import MOCK_ENTERPRISE_DIR, PROCESSED_DIR, read_csv, write_csv


def _growth(current: float, previous: float) -> float:
    return (current - previous) / previous if previous else 0.0


def _level_from_count(count: int) -> str:
    if count >= 3:
        return "红色"
    if count >= 1:
        return "黄色"
    return "绿色"


def build_distribution_features() -> list[dict[str, object]]:
    sales = read_csv(MOCK_ENTERPRISE_DIR / "distribution_sales.csv")
    payments = read_csv(MOCK_ENTERPRISE_DIR / "distribution_payments.csv")
    snapshots = read_csv(MOCK_ENTERPRISE_DIR / "distribution_ar_snapshot.csv")
    as_of_date = max(date.fromisoformat(row["snapshot_date"]) for row in snapshots)
    recent_start = as_of_date - timedelta(days=27)
    previous_start = as_of_date - timedelta(days=55)

    sales_amounts: dict[str, dict[str, float]] = defaultdict(
        lambda: {"recent": 0.0, "previous": 0.0}
    )
    payment_amounts: dict[str, float] = defaultdict(float)
    customer_metadata: dict[str, dict[str, str]] = {}
    for row in sales:
        sale_date = date.fromisoformat(row["sale_date"])
        customer_id = row["customer_id"]
        customer_metadata[customer_id] = {
            "region": row["region"],
            "channel_tier": row["channel_tier"],
        }
        amount = float(row["sales_amount"])
        if recent_start <= sale_date <= as_of_date:
            sales_amounts[customer_id]["recent"] += amount
        elif previous_start <= sale_date < recent_start:
            sales_amounts[customer_id]["previous"] += amount
    for row in payments:
        payment_date = date.fromisoformat(row["payment_date"])
        if recent_start <= payment_date <= as_of_date:
            payment_amounts[row["customer_id"]] += float(row["payment_amount"])

    snapshots_by_customer: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in snapshots:
        snapshots_by_customer[row["customer_id"]].append(row)
    results: list[dict[str, object]] = []
    risk_rank = {"绿色": 0, "黄色": 1, "红色": 2}
    for customer_id, customer_snapshots in snapshots_by_customer.items():
        customer_snapshots.sort(key=lambda row: row["snapshot_date"])
        latest = customer_snapshots[-1]
        comparison = customer_snapshots[-5] if len(customer_snapshots) >= 5 else customer_snapshots[0]
        recent_sales = sales_amounts[customer_id]["recent"]
        previous_sales = sales_amounts[customer_id]["previous"]
        sales_growth = _growth(recent_sales, previous_sales)
        recent_payments = payment_amounts[customer_id]
        payment_coverage = recent_payments / recent_sales if recent_sales else 0.0
        receivable = float(latest["receivable_balance"])
        previous_receivable = float(comparison["receivable_balance"])
        receivable_growth = _growth(receivable, previous_receivable)
        overdue = float(latest["overdue_balance"])
        overdue_ratio = overdue / receivable if receivable else 0.0
        max_overdue_days = int(float(latest["max_overdue_days"]))

        operating_evidence: list[str] = []
        if sales_growth < -0.20:
            operating_evidence.append(f"近4周销售额下降{abs(sales_growth):.1%}")
            operating_level = "红色" if sales_growth < -0.25 else "黄色"
        elif sales_growth < -0.10:
            operating_evidence.append(f"近4周销售额下降{abs(sales_growth):.1%}")
            operating_level = "黄色"
        else:
            operating_level = "绿色"

        receivable_evidence: list[str] = []
        if overdue_ratio > 0.20:
            receivable_evidence.append(f"逾期应收占比{overdue_ratio:.1%}>20%")
        if max_overdue_days > 30:
            receivable_evidence.append(f"最大逾期{max_overdue_days}天>30天")
        if receivable_growth > 0.20:
            receivable_evidence.append(f"近4期应收余额上升{receivable_growth:.1%}>20%")
        if payment_coverage < 0.70:
            receivable_evidence.append(f"近4周回款覆盖率{payment_coverage:.1%}<70%")
        if overdue_ratio > 0.50 or max_overdue_days > 60:
            receivable_level = "红色"
        else:
            receivable_level = _level_from_count(len(receivable_evidence))
        overall_level = max(
            (operating_level, receivable_level), key=lambda value: risk_rank[value]
        )
        health_score = max(
            0.0,
            100.0
            - max(0.0, -sales_growth) * 35
            - overdue_ratio * 35
            - max(0.0, receivable_growth) * 20
            - max(0.0, 0.70 - payment_coverage) * 20,
        )
        metadata = customer_metadata[customer_id]
        results.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "customer_id": customer_id,
                "region": metadata["region"],
                "channel_tier": metadata["channel_tier"],
                "recent_4w_sales_amount": round(recent_sales, 2),
                "previous_4w_sales_amount": round(previous_sales, 2),
                "sales_growth_ratio": round(sales_growth, 4),
                "recent_4w_payment_amount": round(recent_payments, 2),
                "payment_coverage_ratio": round(payment_coverage, 4),
                "receivable_balance": round(receivable, 2),
                "overdue_balance": round(overdue, 2),
                "overdue_ratio": round(overdue_ratio, 4),
                "receivable_growth_ratio": round(receivable_growth, 4),
                "max_overdue_days": max_overdue_days,
                "operating_risk_level": operating_level,
                "operating_evidence": "；".join(operating_evidence) or "未触发经营下滑规则",
                "receivable_risk_level": receivable_level,
                "receivable_evidence": "；".join(receivable_evidence) or "未触发应收回款规则",
                "overall_risk_level": overall_level,
                "health_score_prototype": round(health_score, 1),
                "threshold_status": "prototype_waiting_enterprise_calibration",
                "data_source": "synthetic_enterprise_mock",
            }
        )
    results.sort(key=lambda row: str(row["customer_id"]))
    write_csv(PROCESSED_DIR / "distribution_customer_features.csv", list(results[0]), results)
    return results


if __name__ == "__main__":
    build_distribution_features()
