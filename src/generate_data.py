from __future__ import annotations

import math
import random
from datetime import date, timedelta

from common import SYNTHETIC_DIR, ensure_directories, write_csv


AS_OF_DATE = date(2026, 8, 3)
RANDOM_SEED = 20260803

CUSTOMERS = [
    {
        "customer_id": "C001",
        "customer_name": "稳健数科",
        "region": "华东",
        "channel_tier": "一级",
        "cooperation_months": 72,
        "credit_limit": 700000,
        "credit_term_days": 30,
        "data_source": "synthetic",
    },
    {
        "customer_id": "C002",
        "customer_name": "华南智联",
        "region": "华南",
        "channel_tier": "二级",
        "cooperation_months": 48,
        "credit_limit": 600000,
        "credit_term_days": 45,
        "data_source": "synthetic",
    },
    {
        "customer_id": "C003",
        "customer_name": "东启科技",
        "region": "华东",
        "channel_tier": "二级",
        "cooperation_months": 30,
        "credit_limit": 250000,
        "credit_term_days": 30,
        "data_source": "synthetic",
    },
]

PRODUCTS = [
    {"sku_id": "S001", "sku_name": "企业路由器 R1", "brand": "A品牌", "category": "网络设备", "launch_date": "2025-01-06", "eol_date": "", "unit_price": 1800, "data_source": "synthetic"},
    {"sku_id": "S002", "sku_name": "千兆交换机 S8", "brand": "A品牌", "category": "网络设备", "launch_date": "2024-09-02", "eol_date": "", "unit_price": 1200, "data_source": "synthetic"},
    {"sku_id": "S003", "sku_name": "商用终端 T5", "brand": "A品牌", "category": "智能终端", "launch_date": "2025-03-03", "eol_date": "", "unit_price": 2600, "data_source": "synthetic"},
    {"sku_id": "S004", "sku_name": "存储节点 N2", "brand": "A品牌", "category": "存储", "launch_date": "2024-11-04", "eol_date": "", "unit_price": 3200, "data_source": "synthetic"},
    {"sku_id": "S005", "sku_name": "安全网关 G3", "brand": "A品牌", "category": "安全", "launch_date": "2025-05-05", "eol_date": "", "unit_price": 2800, "data_source": "synthetic"},
]

ASSORTMENT = {
    "C001": {"S001": 20, "S002": 16},
    "C002": {"S002": 24, "S003": 18},
    "C003": {"S003": 20, "S004": 15, "S005": 12},
}


def _sales_multiplier(customer_id: str, week_index: int) -> float:
    if customer_id == "C002" and week_index >= 12:
        return 0.95 - 0.055 * (week_index - 12)
    if customer_id == "C003" and week_index >= 12:
        return 0.90 - 0.050 * (week_index - 12)
    return 1.0


def generate() -> None:
    ensure_directories()
    rng = random.Random(RANDOM_SEED)
    product_prices = {row["sku_id"]: row["unit_price"] for row in PRODUCTS}
    customer_terms = {row["customer_id"]: row["credit_term_days"] for row in CUSTOMERS}
    weeks = [AS_OF_DATE - timedelta(weeks=offset) for offset in reversed(range(24))]

    orders: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    invoices: list[dict[str, object]] = []
    payments: list[dict[str, object]] = []
    inventory = {
        (customer_id, sku_id): base * 4
        for customer_id, products in ASSORTMENT.items()
        for sku_id, base in products.items()
    }

    order_number = 1
    payment_number = 1
    for week_index, week_date in enumerate(weeks):
        for customer_id, products in ASSORTMENT.items():
            region_factor = 1.03 if customer_id in {"C001", "C003"} else 0.98
            seasonal_factor = 1.0 + 0.05 * math.sin(week_index / 3.0)
            promotion_factor = 1.15 if rng.random() < 0.08 else 1.0
            for sku_id, base_demand in products.items():
                noise = rng.uniform(0.94, 1.06)
                quantity = max(
                    1,
                    round(
                        base_demand
                        * region_factor
                        * seasonal_factor
                        * promotion_factor
                        * _sales_multiplier(customer_id, week_index)
                        * noise
                    ),
                )
                price = int(product_prices[sku_id])
                revenue = quantity * price
                order_id = f"O{order_number:04d}"
                invoice_id = f"I{order_number:04d}"
                orders.append(
                    {
                        "order_id": order_id,
                        "order_date": week_date.isoformat(),
                        "customer_id": customer_id,
                        "sku_id": sku_id,
                        "quantity": quantity,
                        "revenue": revenue,
                        "gross_profit": round(revenue * 0.075, 2),
                        "return_flag": 0,
                        "data_source": "synthetic",
                    }
                )

                extra_arrival = 0
                if customer_id == "C002" and week_index >= 14:
                    extra_arrival = round(base_demand * 0.65)
                elif customer_id == "C003" and week_index >= 14:
                    extra_arrival = round(base_demand * 0.85)
                else:
                    extra_arrival = rng.randint(-2, 2)
                arrival = max(0, quantity + extra_arrival)
                key = (customer_id, sku_id)
                inventory[key] = max(0, inventory[key] + arrival - quantity)
                coverage = inventory[key] / max(quantity, 1)
                aging_pressure = max(0.0, coverage - 5.0) * 9.0
                if customer_id in {"C002", "C003"} and week_index >= 14:
                    aging_pressure += (week_index - 13) * 6.0
                inventory_rows.append(
                    {
                        "snapshot_date": week_date.isoformat(),
                        "customer_id": customer_id,
                        "sku_id": sku_id,
                        "on_hand_qty": inventory[key],
                        "in_transit_qty": max(0, round(arrival * 0.15)),
                        "avg_inventory_age_days": round(28 + aging_pressure, 1),
                        "data_source": "synthetic",
                    }
                )

                due_date = week_date + timedelta(days=int(customer_terms[customer_id]))
                payment_date: date | None
                if customer_id == "C003" and week_index >= 10 and order_number % 2 == 0:
                    payment_date = None
                elif customer_id == "C003" and week_index >= 10:
                    candidate = due_date + timedelta(days=12 + week_index - 10)
                    payment_date = candidate if candidate <= AS_OF_DATE else None
                elif customer_id == "C001" and week_index == 17 and sku_id == "S001":
                    payment_date = due_date + timedelta(days=5)
                else:
                    payment_date = due_date - timedelta(days=rng.randint(0, 3))
                if payment_date is not None and payment_date <= AS_OF_DATE:
                    open_amount = 0
                    payments.append(
                        {
                            "payment_id": f"P{payment_number:04d}",
                            "invoice_id": invoice_id,
                            "payment_date": payment_date.isoformat(),
                            "payment_amount": revenue,
                            "data_source": "synthetic",
                        }
                    )
                    payment_number += 1
                else:
                    open_amount = revenue
                invoices.append(
                    {
                        "invoice_id": invoice_id,
                        "order_id": order_id,
                        "customer_id": customer_id,
                        "issue_date": week_date.isoformat(),
                        "due_date": due_date.isoformat(),
                        "invoice_amount": revenue,
                        "open_amount": open_amount,
                        "dispute_flag": 0,
                        "data_source": "synthetic",
                    }
                )
                order_number += 1

    write_csv(SYNTHETIC_DIR / "customer.csv", list(CUSTOMERS[0]), CUSTOMERS)
    write_csv(SYNTHETIC_DIR / "product.csv", list(PRODUCTS[0]), PRODUCTS)
    write_csv(SYNTHETIC_DIR / "sales_order.csv", list(orders[0]), orders)
    write_csv(SYNTHETIC_DIR / "inventory_snapshot.csv", list(inventory_rows[0]), inventory_rows)
    write_csv(SYNTHETIC_DIR / "invoice.csv", list(invoices[0]), invoices)
    write_csv(SYNTHETIC_DIR / "payment.csv", list(payments[0]), payments)
    write_csv(
        SYNTHETIC_DIR / "intervention.csv",
        [
            "intervention_id",
            "risk_event_id",
            "customer_id",
            "action_type",
            "owner",
            "status",
            "created_at",
            "completed_at",
            "result",
            "data_source",
        ],
        [],
    )


if __name__ == "__main__":
    generate()
