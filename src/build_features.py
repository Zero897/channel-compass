from __future__ import annotations

from collections import defaultdict
from datetime import date

from common import PROCESSED_DIR, SYNTHETIC_DIR, read_csv, write_csv
from health_score import calculate_dimension_scores, calculate_health_score
from inventory_risk import evaluate_inventory_risk
from payment_risk import evaluate_payment_risk


AS_OF_DATE = date(2026, 8, 3)


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build() -> list[dict[str, object]]:
    customers = read_csv(SYNTHETIC_DIR / "customer.csv")
    orders = read_csv(SYNTHETIC_DIR / "sales_order.csv")
    inventory = read_csv(SYNTHETIC_DIR / "inventory_snapshot.csv")
    products = read_csv(SYNTHETIC_DIR / "product.csv")
    invoices = read_csv(SYNTHETIC_DIR / "invoice.csv")
    payments = read_csv(SYNTHETIC_DIR / "payment.csv")
    payment_by_invoice = {row["invoice_id"]: row for row in payments}
    price_by_sku = {row["sku_id"]: float(row["unit_price"]) for row in products}

    weekly_sales: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in orders:
        weekly_sales[row["customer_id"]][row["order_date"]] += _float(row, "quantity")

    latest_snapshot = max(row["snapshot_date"] for row in inventory)
    latest_inventory: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in inventory:
        if row["snapshot_date"] == latest_snapshot:
            latest_inventory[row["customer_id"]].append(row)

    customer_invoices: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in invoices:
        customer_invoices[row["customer_id"]].append(row)

    results: list[dict[str, object]] = []
    for customer in customers:
        customer_id = customer["customer_id"]
        sales_by_week = weekly_sales[customer_id]
        week_dates = sorted(sales_by_week)
        previous_sales = sum(sales_by_week[item] for item in week_dates[-8:-4])
        recent_sales = sum(sales_by_week[item] for item in week_dates[-4:])
        sales_growth = (recent_sales - previous_sales) / previous_sales if previous_sales else 0.0
        recent_weekly_sales = recent_sales / 4.0

        inventory_items = latest_inventory[customer_id]
        on_hand = sum(_float(row, "on_hand_qty") for row in inventory_items)
        inventory_value = sum(
            _float(row, "on_hand_qty") * price_by_sku[row["sku_id"]]
            for row in inventory_items
        )
        coverage = on_hand / max(recent_weekly_sales, 1.0)
        weighted_age_numerator = sum(
            _float(row, "on_hand_qty") * _float(row, "avg_inventory_age_days")
            for row in inventory_items
        )
        average_age = weighted_age_numerator / on_hand if on_hand else 0.0

        all_invoices = customer_invoices[customer_id]
        total_open = sum(_float(row, "open_amount") for row in all_invoices)
        overdue_open = sum(
            _float(row, "open_amount")
            for row in all_invoices
            if row["open_amount"] != "0" and date.fromisoformat(row["due_date"]) < AS_OF_DATE
        )
        overdue_ratio = overdue_open / total_open if total_open else 0.0
        matured = [row for row in all_invoices if date.fromisoformat(row["due_date"]) <= AS_OF_DATE]
        overdue_days: list[tuple[str, float]] = []
        on_time_count = 0
        for invoice in matured:
            due_date = date.fromisoformat(invoice["due_date"])
            payment = payment_by_invoice.get(invoice["invoice_id"])
            if payment:
                payment_date = date.fromisoformat(payment["payment_date"])
                delay = max(0, (payment_date - due_date).days)
                if payment_date <= due_date:
                    on_time_count += 1
            else:
                delay = max(0, (AS_OF_DATE - due_date).days)
            overdue_days.append((invoice["issue_date"], float(delay)))
        on_time_rate = on_time_count / len(matured) if matured else 1.0
        overdue_days.sort(key=lambda item: item[0])
        recent_overdue = _average([item[1] for item in overdue_days[-6:]])
        previous_overdue = _average([item[1] for item in overdue_days[-12:-6]])
        credit_utilization = total_open / float(customer["credit_limit"])

        numeric_features = {
            "sales_growth_ratio": sales_growth,
            "inventory_coverage_weeks": coverage,
            "average_inventory_age_days": average_age,
            "overdue_amount_ratio": overdue_ratio,
            "on_time_payment_rate": on_time_rate,
            "credit_utilization_ratio": credit_utilization,
            "recent_average_overdue_days": recent_overdue,
            "previous_average_overdue_days": previous_overdue,
            "cooperation_months": float(customer["cooperation_months"]),
        }
        inventory_level, inventory_evidence = evaluate_inventory_risk(
            coverage, average_age, sales_growth
        )
        payment_level, payment_evidence = evaluate_payment_risk(
            overdue_ratio,
            on_time_rate,
            credit_utilization,
            recent_overdue,
            previous_overdue,
        )
        dimension_scores = calculate_dimension_scores(numeric_features)
        health_score, health_level = calculate_health_score(dimension_scores)
        rule_rank = {"绿色": 0, "黄色": 1, "红色": 2}
        overall_level = max(
            (health_level, inventory_level, payment_level), key=lambda item: rule_rank[item]
        )
        result: dict[str, object] = {
            "customer_id": customer_id,
            "customer_name": customer["customer_name"],
            "region": customer["region"],
            "channel_tier": customer["channel_tier"],
            **{key: round(value, 4) for key, value in numeric_features.items()},
            **dimension_scores,
            "health_score": health_score,
            "health_level": health_level,
            "inventory_risk_level": inventory_level,
            "inventory_evidence": "；".join(inventory_evidence) or "未触发库存规则",
            "payment_risk_level": payment_level,
            "payment_evidence": "；".join(payment_evidence) or "未触发回款规则",
            "overall_risk_level": overall_level,
            "open_amount": round(total_open, 2),
            "overdue_open_amount": round(overdue_open, 2),
            "latest_inventory_value": round(inventory_value, 2),
            "data_source": "synthetic",
        }
        results.append(result)

    write_csv(PROCESSED_DIR / "customer_features.csv", list(results[0]), results)
    return results


if __name__ == "__main__":
    build()
