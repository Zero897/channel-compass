from __future__ import annotations

from collections import defaultdict
from datetime import date

from common import MOCK_ENTERPRISE_DIR, SYNTHETIC_DIR, ensure_directories, read_csv, write_csv


def generate_mock_enterprise_tables() -> None:
    ensure_directories()
    customers = {row["customer_id"]: row for row in read_csv(SYNTHETIC_DIR / "customer.csv")}
    orders = read_csv(SYNTHETIC_DIR / "sales_order.csv")
    invoices = read_csv(SYNTHETIC_DIR / "invoice.csv")
    payments = read_csv(SYNTHETIC_DIR / "payment.csv")

    sales_rows = [
        {
            "sale_id": row["order_id"],
            "sale_date": row["order_date"],
            "customer_id": row["customer_id"],
            "sales_amount": row["revenue"],
            "sku_id": row["sku_id"],
            "quantity": row["quantity"],
            "region": customers[row["customer_id"]]["region"],
            "channel_tier": customers[row["customer_id"]]["channel_tier"],
            "data_source": "synthetic_enterprise_mock",
        }
        for row in orders
    ]

    invoice_by_id = {row["invoice_id"]: row for row in invoices}
    payment_rows = [
        {
            "payment_id": row["payment_id"],
            "payment_date": row["payment_date"],
            "customer_id": invoice_by_id[row["invoice_id"]]["customer_id"],
            "payment_amount": row["payment_amount"],
            "related_document_id": row["invoice_id"],
            "data_source": "synthetic_enterprise_mock",
        }
        for row in payments
    ]

    payment_date_by_invoice = {
        row["invoice_id"]: date.fromisoformat(row["payment_date"]) for row in payments
    }
    invoices_by_customer: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in invoices:
        invoices_by_customer[row["customer_id"]].append(row)
    snapshot_dates = sorted({date.fromisoformat(row["order_date"]) for row in orders})
    ar_rows: list[dict[str, object]] = []
    for snapshot_date in snapshot_dates:
        for customer_id in customers:
            open_invoices: list[dict[str, str]] = []
            for invoice in invoices_by_customer[customer_id]:
                issue_date = date.fromisoformat(invoice["issue_date"])
                payment_date = payment_date_by_invoice.get(invoice["invoice_id"])
                if issue_date <= snapshot_date and (
                    payment_date is None or payment_date > snapshot_date
                ):
                    open_invoices.append(invoice)
            receivable_balance = sum(float(row["invoice_amount"]) for row in open_invoices)
            overdue_invoices = [
                row
                for row in open_invoices
                if date.fromisoformat(row["due_date"]) < snapshot_date
            ]
            overdue_balance = sum(float(row["invoice_amount"]) for row in overdue_invoices)
            max_overdue_days = max(
                (
                    (snapshot_date - date.fromisoformat(row["due_date"])).days
                    for row in overdue_invoices
                ),
                default=0,
            )
            ar_rows.append(
                {
                    "snapshot_date": snapshot_date.isoformat(),
                    "customer_id": customer_id,
                    "receivable_balance": round(receivable_balance, 2),
                    "overdue_balance": round(overdue_balance, 2),
                    "max_overdue_days": max_overdue_days,
                    "open_document_count": len(open_invoices),
                    "data_source": "synthetic_enterprise_mock",
                }
            )

    write_csv(MOCK_ENTERPRISE_DIR / "distribution_sales.csv", list(sales_rows[0]), sales_rows)
    write_csv(
        MOCK_ENTERPRISE_DIR / "distribution_payments.csv",
        list(payment_rows[0]),
        payment_rows,
    )
    write_csv(
        MOCK_ENTERPRISE_DIR / "distribution_ar_snapshot.csv",
        list(ar_rows[0]),
        ar_rows,
    )


if __name__ == "__main__":
    generate_mock_enterprise_tables()
