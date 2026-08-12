from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import groupby
from pathlib import Path
from typing import Iterable, Iterator

from project_paths import portable_path


csv.field_size_limit(2_147_483_647)

OUTCOME_HORIZON_DAYS = 120
TRAIN_END = "2025-03-31"
VALIDATION_START = "2025-08-01"
VALIDATION_END = "2025-09-30"
TEST_START = "2026-02-01"
TEST_END = "2026-03-31"
LATEST_AR_SNAPSHOT = "2026-07-31"

LABEL_FIELDS = [
    "order_id",
    "customer_id",
    "order_date",
    "dataset_split",
    "label",
    "label_status",
    "label_evidence_date",
    "outcome_horizon_date",
    "outcome_snapshot_date",
    "label_matured",
    "has_payment_record",
    "ever_overdue_payment",
    "ever_overdue_ar",
    "present_in_latest_ar",
]

FEATURE_FIELDS = [
    "order_id",
    "customer_id",
    "order_date",
    "dataset_split",
    "label",
    "label_status",
    "customer_type",
    "region",
    "province",
    "order_amount",
    "cost_amount",
    "gross_margin_ratio",
    "quantity",
    "line_count",
    "sku_count",
    "product_line_count",
    "return_line_count",
    "return_amount_abs",
    "price_protection_amount",
    "vendor_rebate_amount",
    "cash_discount_amount",
    "avg_inventory_age",
    "payment_term_days",
    "prior_order_count",
    "prior_sales_amount",
    "prior_sales_30d",
    "prior_sales_previous_30d",
    "prior_sales_90d",
    "prior_sales_180d",
    "prior_orders_30d",
    "prior_orders_90d",
    "prior_return_ratio_180d",
    "prior_sales_30d_growth",
    "prior_payment_count",
    "prior_overdue_payment_count",
    "prior_overdue_payment_rate",
    "prior_payment_amount",
    "prior_overdue_payment_amount",
    "prior_overdue_amount_ratio",
    "prior_avg_payment_age_days",
    "latest_prior_ar_amount",
    "latest_prior_overdue_ar_amount",
    "latest_prior_overdue_ar_ratio",
    "latest_prior_overdue_30_amount",
    "latest_prior_overdue_60_amount",
    "latest_prior_max_overdue_days",
    "prior_ar_snapshot_age_days",
    "prior_ar_missing",
    "prior_extension_count",
    "prior_extension_amount",
]

MODEL_FEATURE_FIELDS = FEATURE_FIELDS[6:]
FORBIDDEN_SOURCE_FIELDS = {
    "回款日期",
    "是否超期",
    "超期天数",
    "超期利息金额",
    "收款编号",
    "最终承诺还款日期",
}


def _rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV没有表头：{path}")
        for row in reader:
            yield {key: (value or "").strip() for key, value in row.items()}


def _identifier(value: str) -> str:
    value = value.strip()
    if value.endswith(".0") and value[:-2].isdigit():
        return value[:-2]
    return value


def _number(value: str) -> float:
    if value in {"", "无", "nan", "NaN", "None"}:
        return 0.0
    return float(value.replace(",", ""))


def _flag(value: str) -> int:
    return int(value.strip().upper() in {"Y", "YES", "是", "1", "TRUE"})


def _payment_term_days(value: str) -> int:
    if "立即" in value:
        return 0
    match = re.search(r"(\d+)\s*天", value)
    return int(match.group(1)) if match else -1


def _split(order_date: str) -> str:
    if order_date <= TRAIN_END:
        return "train"
    if VALIDATION_START <= order_date <= VALIDATION_END:
        return "validation"
    if TEST_START <= order_date <= TEST_END:
        return "test"
    if order_date > TEST_END:
        return "scoring_holdout"
    return "embargo"


def _minimum_date(*values: str | None) -> str:
    present = [value for value in values if value]
    return min(present) if present else ""


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _round(value: float) -> float:
    return round(value, 6)


def _batches(rows: Iterable[tuple[object, ...]], size: int = 10_000) -> Iterator[list[tuple[object, ...]]]:
    batch: list[tuple[object, ...]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_project_contracts(path: Path) -> set[str]:
    return {
        _identifier(row["合同编号"])
        for row in _rows(path)
        if _identifier(row["合同编号"])
    }


def _sales_records(path: Path, project_contracts: set[str]) -> Iterator[tuple[object, ...]]:
    for row in _rows(path):
        amount = _number(row["销售金额_折扣后_含税"])
        quantity = _number(row["数量"])
        contract = _identifier(row["合同号"])
        yield (
            _identifier(row["销售订单号"]),
            row["出库日期"][:10],
            _identifier(row["客户编号"]),
            row["客户类型"],
            row["客户大区"],
            row["客户所在省份"],
            _identifier(row["物料编码"]),
            row["产品线名称"],
            _payment_term_days(row["付款条件"]),
            quantity,
            amount,
            _number(row["出库成本金额"]),
            _number(row["价保"]),
            _number(row["厂商返利"]),
            _number(row["现金折扣"]),
            _number(row["在库库龄"]),
            int(amount < 0 or quantity < 0),
            abs(min(amount, 0.0)),
            int(contract in project_contracts),
        )


def _payment_records(path: Path) -> Iterator[tuple[object, ...]]:
    for row in _rows(path):
        yield (
            _identifier(row["销售订单号"]),
            _identifier(row["客户编号"]),
            row["回款日期"][:10],
            _flag(row["是否超期"]),
            _number(row["回款金额"]),
            _number(row["回款账龄"]),
        )


def _ar_records(path: Path) -> Iterator[tuple[object, ...]]:
    for row in _rows(path):
        order_id = _identifier(row["销售订单号"])
        customer_id = _identifier(row["客户编号"])
        if not order_id or not customer_id:
            continue
        yield (
            order_id,
            customer_id,
            row["快照时间"][:10],
            _flag(row["是否超期"]),
            _number(row["应收金额"]),
            _number(row["超期应收金额"]),
            _number(row["超期30天以上金额"]),
            _number(row["超期60天以上金额"]),
            _number(row["超期天数"]),
        )


def _extension_records(path: Path) -> Iterator[tuple[object, ...]]:
    for row in _rows(path):
        customer_id = _identifier(row["客户编号"])
        if customer_id:
            yield customer_id, row["快照时间"][:10], _number(row["应收金额"])


def _prepare_database(connection: sqlite3.Connection, source: Path) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE sales_line (
            order_id TEXT, order_date TEXT, customer_id TEXT, customer_type TEXT,
            region TEXT, province TEXT, sku_id TEXT, product_line TEXT,
            payment_term_days INTEGER, quantity REAL, sales_amount REAL,
            cost_amount REAL, price_protection REAL, vendor_rebate REAL,
            cash_discount REAL, inventory_age REAL, return_line INTEGER,
            return_amount REAL, is_project INTEGER
        );
        CREATE TABLE payment_fact (
            order_id TEXT, customer_id TEXT, payment_date TEXT, overdue INTEGER,
            payment_amount REAL, payment_age REAL
        );
        CREATE TABLE ar_fact (
            order_id TEXT, customer_id TEXT, snapshot_date TEXT, overdue INTEGER,
            ar_amount REAL, overdue_amount REAL, overdue_30 REAL,
            overdue_60 REAL, overdue_days REAL
        );
        CREATE TABLE extension_fact (customer_id TEXT, event_date TEXT, amount REAL);
        """
    )
    project_contracts = _load_project_contracts(source / "增值合同签约明细.csv")
    loaders = (
        (
            "INSERT INTO sales_line VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            _sales_records(source / "销售流水.csv", project_contracts),
        ),
        ("INSERT INTO payment_fact VALUES (?,?,?,?,?,?)", _payment_records(source / "业务回款明细.csv")),
        ("INSERT INTO ar_fact VALUES (?,?,?,?,?,?,?,?,?)", _ar_records(source / "应收快照_月末24期.csv")),
        ("INSERT INTO extension_fact VALUES (?,?,?)", _extension_records(source / "展期记录.csv")),
    )
    for statement, records in loaders:
        for batch in _batches(records):
            connection.executemany(statement, batch)
        connection.commit()
    connection.executescript(
        f"""
        CREATE TABLE sales_order AS
        SELECT order_id, MIN(order_date) AS order_date, MIN(customer_id) AS customer_id,
               MIN(customer_type) AS customer_type, MIN(region) AS region,
               MIN(province) AS province, SUM(sales_amount) AS order_amount,
               SUM(cost_amount) AS cost_amount, SUM(quantity) AS quantity,
               COUNT(*) AS line_count, COUNT(DISTINCT sku_id) AS sku_count,
               COUNT(DISTINCT product_line) AS product_line_count,
               SUM(return_line) AS return_line_count, SUM(return_amount) AS return_amount_abs,
               SUM(price_protection) AS price_protection_amount,
               SUM(vendor_rebate) AS vendor_rebate_amount,
               SUM(cash_discount) AS cash_discount_amount,
               AVG(inventory_age) AS avg_inventory_age,
               MAX(payment_term_days) AS payment_term_days,
               MAX(is_project) AS is_project
        FROM sales_line
        WHERE order_id <> '' AND customer_id <> ''
        GROUP BY order_id;

        CREATE TABLE snapshot_calendar AS
        SELECT DISTINCT snapshot_date FROM ar_fact WHERE snapshot_date <> '';
        CREATE INDEX snapshot_calendar_date_idx ON snapshot_calendar(snapshot_date);
        CREATE INDEX payment_order_date_idx ON payment_fact(order_id, payment_date);
        CREATE INDEX ar_order_snapshot_idx ON ar_fact(order_id, snapshot_date);

        CREATE TABLE order_outcome AS
        SELECT s.*,
               date(s.order_date, '+{OUTCOME_HORIZON_DAYS} day') AS outcome_horizon_date,
               (
                   SELECT MIN(c.snapshot_date)
                   FROM snapshot_calendar c
                   WHERE c.snapshot_date >= date(s.order_date, '+{OUTCOME_HORIZON_DAYS} day')
               ) AS outcome_snapshot_date
        FROM sales_order s;

        CREATE TABLE payment_order AS
        SELECT s.order_id,
               MAX(CASE WHEN p.payment_date BETWEEN s.order_date AND s.outcome_horizon_date THEN 1 ELSE 0 END) AS has_payment,
               MAX(CASE WHEN p.payment_date BETWEEN s.order_date AND s.outcome_horizon_date THEN p.overdue ELSE 0 END) AS ever_overdue_payment,
               MIN(CASE WHEN p.overdue=1 AND p.payment_date BETWEEN s.order_date AND s.outcome_horizon_date THEN p.payment_date END) AS first_overdue_payment_date,
               MAX(CASE WHEN p.payment_date BETWEEN s.order_date AND s.outcome_horizon_date THEN p.payment_date END) AS last_payment_date
        FROM order_outcome s
        LEFT JOIN payment_fact p ON p.order_id=s.order_id
        GROUP BY s.order_id;

        CREATE TABLE ar_order AS
        SELECT s.order_id,
               MAX(CASE WHEN a.snapshot_date BETWEEN s.order_date AND s.outcome_horizon_date THEN a.overdue ELSE 0 END) AS ever_overdue_ar,
               MIN(CASE WHEN a.overdue=1 AND a.snapshot_date BETWEEN s.order_date AND s.outcome_horizon_date THEN a.snapshot_date END) AS first_overdue_ar_date,
               MAX(CASE WHEN a.snapshot_date=s.outcome_snapshot_date AND a.ar_amount>0 THEN 1 ELSE 0 END) AS present_at_outcome_snapshot,
               MAX(CASE WHEN a.snapshot_date='{LATEST_AR_SNAPSHOT}' AND a.ar_amount>0 THEN 1 ELSE 0 END) AS present_in_latest_ar
        FROM order_outcome s
        LEFT JOIN ar_fact a ON a.order_id=s.order_id
        GROUP BY s.order_id;

        CREATE TABLE payment_daily AS
        SELECT customer_id, payment_date, COUNT(*) AS payment_count,
               SUM(overdue) AS overdue_count, SUM(payment_amount) AS payment_amount,
               SUM(CASE WHEN overdue=1 THEN payment_amount ELSE 0 END) AS overdue_amount,
               SUM(payment_age) AS payment_age_sum
        FROM payment_fact GROUP BY customer_id, payment_date;

        CREATE TABLE ar_daily AS
        SELECT customer_id, snapshot_date, SUM(ar_amount) AS ar_amount,
               SUM(overdue_amount) AS overdue_amount, SUM(overdue_30) AS overdue_30,
               SUM(overdue_60) AS overdue_60, MAX(overdue_days) AS max_overdue_days
        FROM ar_fact GROUP BY customer_id, snapshot_date;

        CREATE TABLE extension_daily AS
        SELECT customer_id, event_date, COUNT(*) AS extension_count, SUM(amount) AS extension_amount
        FROM extension_fact GROUP BY customer_id, event_date;

        CREATE INDEX sales_order_date_idx ON sales_order(order_date, order_id);
        """
    )


def _events_by_customer(
    connection: sqlite3.Connection, query: str
) -> dict[str, list[tuple[object, ...]]]:
    result: dict[str, list[tuple[object, ...]]] = defaultdict(list)
    for row in connection.execute(query):
        result[str(row[0])].append(tuple(row[1:]))
    return result


@dataclass
class SalesHistory:
    windows: dict[int, deque[tuple[date, float, float]]] = field(
        default_factory=lambda: {days: deque() for days in (30, 60, 90, 180)}
    )
    sums: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    returns: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    total_count: int = 0
    total_sales: float = 0.0

    def advance(self, current: date) -> None:
        for days, events in self.windows.items():
            threshold = current - timedelta(days=days)
            while events and events[0][0] < threshold:
                _, amount, return_amount = events.popleft()
                self.sums[days] -= amount
                self.returns[days] -= return_amount
                self.counts[days] -= 1

    def add(self, event_date: date, amount: float, return_amount: float) -> None:
        self.total_count += 1
        self.total_sales += amount
        for days, events in self.windows.items():
            events.append((event_date, amount, return_amount))
            self.sums[days] += amount
            self.returns[days] += return_amount
            self.counts[days] += 1


@dataclass
class EventCursor:
    events: list[tuple[object, ...]]
    index: int = 0
    totals: list[float] = field(default_factory=list)

    def advance(self, current: str, width: int) -> list[float]:
        if not self.totals:
            self.totals = [0.0] * width
        while self.index < len(self.events) and str(self.events[self.index][0]) < current:
            event = self.events[self.index]
            for position in range(width):
                self.totals[position] += float(event[position + 1])
            self.index += 1
        return self.totals


@dataclass
class SnapshotCursor:
    events: list[tuple[object, ...]]
    index: int = 0
    latest: tuple[object, ...] | None = None

    def advance(self, current: str) -> tuple[object, ...] | None:
        while self.index < len(self.events) and str(self.events[self.index][0]) < current:
            self.latest = self.events[self.index]
            self.index += 1
        return self.latest


def _label(row: sqlite3.Row | dict[str, object]) -> tuple[str, str, str]:
    if int(row["is_project"]):
        return "", "excluded_project_business", ""
    if not row["outcome_snapshot_date"]:
        if int(row["present_in_latest_ar"] or 0):
            return "", "insufficient_open_receivable", ""
        return "", "insufficient_unmatured", ""
    payment_overdue = int(row["ever_overdue_payment"] or 0)
    ar_overdue = int(row["ever_overdue_ar"] or 0)
    if payment_overdue or ar_overdue:
        return (
            "1",
            "eligible",
            _minimum_date(row["first_overdue_payment_date"], row["first_overdue_ar_date"]),
        )
    if int(row["has_payment"] or 0) and not int(row["present_at_outcome_snapshot"] or 0):
        return "0", "eligible", str(row["outcome_snapshot_date"])
    if int(row["present_at_outcome_snapshot"] or 0):
        return "", "insufficient_open_receivable", ""
    return "", "insufficient_no_outcome", ""


def _base_query() -> str:
    return """
        SELECT s.*, p.has_payment, p.ever_overdue_payment,
               p.first_overdue_payment_date, p.last_payment_date,
               a.ever_overdue_ar, a.first_overdue_ar_date,
               a.present_at_outcome_snapshot, a.present_in_latest_ar
        FROM order_outcome s
        LEFT JOIN payment_order p USING(order_id)
        LEFT JOIN ar_order a USING(order_id)
        ORDER BY s.order_date, s.order_id
    """


def _write_training_outputs(
    connection: sqlite3.Connection, labels_path: Path, features_path: Path
) -> dict[str, object]:
    payment_events = _events_by_customer(
        connection,
        "SELECT customer_id, payment_date, payment_count, overdue_count, payment_amount, overdue_amount, payment_age_sum FROM payment_daily ORDER BY customer_id, payment_date",
    )
    ar_events = _events_by_customer(
        connection,
        "SELECT customer_id, snapshot_date, ar_amount, overdue_amount, overdue_30, overdue_60, max_overdue_days FROM ar_daily ORDER BY customer_id, snapshot_date",
    )
    extension_events = _events_by_customer(
        connection,
        "SELECT customer_id, event_date, extension_count, extension_amount FROM extension_daily ORDER BY customer_id, event_date",
    )
    sales_history: dict[str, SalesHistory] = defaultdict(SalesHistory)
    payment_cursors = {key: EventCursor(value) for key, value in payment_events.items()}
    ar_cursors = {key: SnapshotCursor(value) for key, value in ar_events.items()}
    extension_cursors = {key: EventCursor(value) for key, value in extension_events.items()}
    status_counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(_base_query())
    with labels_path.open("w", encoding="utf-8-sig", newline="") as label_handle, features_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as feature_handle:
        label_writer = csv.DictWriter(label_handle, fieldnames=LABEL_FIELDS)
        feature_writer = csv.DictWriter(feature_handle, fieldnames=FEATURE_FIELDS)
        label_writer.writeheader()
        feature_writer.writeheader()
        for order_date, day_rows_iter in groupby(rows, key=lambda item: str(item["order_date"])):
            day_rows = list(day_rows_iter)
            current_day = date.fromisoformat(order_date)
            pending_history: list[tuple[str, float, float]] = []
            for row in day_rows:
                customer_id = str(row["customer_id"])
                split = _split(order_date)
                label, status, evidence_date = _label(row)
                status_counts[status] += 1
                split_counts[split][label if label else status] += 1
                label_writer.writerow(
                    {
                        "order_id": row["order_id"],
                        "customer_id": customer_id,
                        "order_date": order_date,
                        "dataset_split": split,
                        "label": label,
                        "label_status": status,
                        "label_evidence_date": evidence_date,
                        "outcome_horizon_date": row["outcome_horizon_date"],
                        "outcome_snapshot_date": row["outcome_snapshot_date"] or "",
                        "label_matured": int(bool(row["outcome_snapshot_date"])),
                        "has_payment_record": int(row["has_payment"] or 0),
                        "ever_overdue_payment": int(row["ever_overdue_payment"] or 0),
                        "ever_overdue_ar": int(row["ever_overdue_ar"] or 0),
                        "present_in_latest_ar": int(row["present_in_latest_ar"] or 0),
                    }
                )
                history = sales_history[customer_id]
                history.advance(current_day)
                payment = payment_cursors.get(customer_id)
                payment_totals = payment.advance(order_date, 5) if payment else [0.0] * 5
                ar = ar_cursors.get(customer_id)
                latest_ar = ar.advance(order_date) if ar else None
                extensions = extension_cursors.get(customer_id)
                extension_totals = extensions.advance(order_date, 2) if extensions else [0.0] * 2
                recent_30 = history.sums[30]
                previous_30 = history.sums[60] - recent_30
                ar_date = str(latest_ar[0]) if latest_ar else ""
                ar_amount = float(latest_ar[1]) if latest_ar else 0.0
                overdue_ar = float(latest_ar[2]) if latest_ar else 0.0
                feature_writer.writerow(
                    {
                        "order_id": row["order_id"],
                        "customer_id": customer_id,
                        "order_date": order_date,
                        "dataset_split": split,
                        "label": label,
                        "label_status": status,
                        "customer_type": row["customer_type"],
                        "region": row["region"],
                        "province": row["province"],
                        "order_amount": _round(float(row["order_amount"] or 0.0)),
                        "cost_amount": _round(float(row["cost_amount"] or 0.0)),
                        "gross_margin_ratio": _round(
                            _ratio(
                                float(row["order_amount"] or 0.0) - float(row["cost_amount"] or 0.0),
                                abs(float(row["order_amount"] or 0.0)),
                            )
                        ),
                        "quantity": _round(float(row["quantity"] or 0.0)),
                        "line_count": int(row["line_count"]),
                        "sku_count": int(row["sku_count"]),
                        "product_line_count": int(row["product_line_count"]),
                        "return_line_count": int(row["return_line_count"]),
                        "return_amount_abs": _round(float(row["return_amount_abs"] or 0.0)),
                        "price_protection_amount": _round(float(row["price_protection_amount"] or 0.0)),
                        "vendor_rebate_amount": _round(float(row["vendor_rebate_amount"] or 0.0)),
                        "cash_discount_amount": _round(float(row["cash_discount_amount"] or 0.0)),
                        "avg_inventory_age": _round(float(row["avg_inventory_age"] or 0.0)),
                        "payment_term_days": int(row["payment_term_days"]),
                        "prior_order_count": history.total_count,
                        "prior_sales_amount": _round(history.total_sales),
                        "prior_sales_30d": _round(recent_30),
                        "prior_sales_previous_30d": _round(previous_30),
                        "prior_sales_90d": _round(history.sums[90]),
                        "prior_sales_180d": _round(history.sums[180]),
                        "prior_orders_30d": history.counts[30],
                        "prior_orders_90d": history.counts[90],
                        "prior_return_ratio_180d": _round(
                            _ratio(history.returns[180], abs(history.sums[180]))
                        ),
                        "prior_sales_30d_growth": _round(
                            _ratio(recent_30 - previous_30, abs(previous_30))
                        ),
                        "prior_payment_count": int(payment_totals[0]),
                        "prior_overdue_payment_count": int(payment_totals[1]),
                        "prior_overdue_payment_rate": _round(
                            _ratio(payment_totals[1], payment_totals[0])
                        ),
                        "prior_payment_amount": _round(payment_totals[2]),
                        "prior_overdue_payment_amount": _round(payment_totals[3]),
                        "prior_overdue_amount_ratio": _round(
                            _ratio(payment_totals[3], abs(payment_totals[2]))
                        ),
                        "prior_avg_payment_age_days": _round(
                            _ratio(payment_totals[4], payment_totals[0])
                        ),
                        "latest_prior_ar_amount": _round(ar_amount),
                        "latest_prior_overdue_ar_amount": _round(overdue_ar),
                        "latest_prior_overdue_ar_ratio": _round(_ratio(overdue_ar, ar_amount)),
                        "latest_prior_overdue_30_amount": _round(float(latest_ar[3]) if latest_ar else 0.0),
                        "latest_prior_overdue_60_amount": _round(float(latest_ar[4]) if latest_ar else 0.0),
                        "latest_prior_max_overdue_days": _round(float(latest_ar[5]) if latest_ar else 0.0),
                        "prior_ar_snapshot_age_days": (current_day - date.fromisoformat(ar_date)).days if ar_date else -1,
                        "prior_ar_missing": int(not latest_ar),
                        "prior_extension_count": int(extension_totals[0]),
                        "prior_extension_amount": _round(extension_totals[1]),
                    }
                )
                pending_history.append(
                    (customer_id, float(row["order_amount"] or 0.0), float(row["return_amount_abs"] or 0.0))
                )
            for customer_id, amount, return_amount in pending_history:
                sales_history[customer_id].add(current_day, amount, return_amount)
    split_summary: dict[str, dict[str, object]] = {}
    for split, counts in split_counts.items():
        positive = counts["1"]
        negative = counts["0"]
        eligible = positive + negative
        split_summary[split] = {
            "positive": positive,
            "negative": negative,
            "eligible": eligible,
            "positive_rate": _round(_ratio(positive, eligible)),
            "other_statuses": {
                key: value for key, value in counts.items() if key not in {"0", "1"}
            },
        }
    return {
        "total_orders": sum(status_counts.values()),
        "label_status_counts": dict(status_counts),
        "splits": split_summary,
    }


def build_training_data(source: Path, processed: Path, report_path: Path) -> dict[str, object]:
    required = [
        "销售流水.csv",
        "业务回款明细.csv",
        "应收快照_月末24期.csv",
        "增值合同签约明细.csv",
        "展期记录.csv",
        "客户授信.csv",
    ]
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"缺少企业数据文件：{', '.join(missing)}")
    processed.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    work_db = processed / "training_build_work.sqlite"
    if work_db.exists():
        work_db.unlink()
    connection = sqlite3.connect(work_db)
    try:
        try:
            _prepare_database(connection, source)
            summary = _write_training_outputs(
                connection,
                processed / "order_labels.csv",
                processed / "order_features.csv",
            )
        finally:
            connection.close()
    finally:
        if work_db.exists():
            work_db.unlink()
    forbidden_intersection = sorted(FORBIDDEN_SOURCE_FIELDS & set(MODEL_FEATURE_FIELDS))
    report = {
        "status": "pass" if not forbidden_intersection else "fail",
        "source": "AFFT模拟数据集",
        "task": "订单出库时预测后续是否发生超期回款",
        "label_rule": {
            "positive": f"出库后{OUTCOME_HORIZON_DAYS}天内出现超期回款或月末超期应收",
            "negative": f"出库后{OUTCOME_HORIZON_DAYS}天内存在回款、未出现超期，且观察期后首个月末快照无未结余额",
            "unknown": "观察期未成熟、期末仍有未结应收或没有足够结果证据，不强行标注",
        },
        "outcome_observation": {
            "horizon_days": OUTCOME_HORIZON_DAYS,
            "maturity_confirmation": "观察期结束后的首个月末应收快照",
            "latest_available_snapshot": LATEST_AR_SNAPSHOT,
        },
        "time_split": {
            "train": f"<= {TRAIN_END}",
            "validation": f"{VALIDATION_START}至{VALIDATION_END}",
            "test": f"{TEST_START}至{TEST_END}",
            "embargo": f"训练、验证、测试窗口之间至少隔离{OUTCOME_HORIZON_DAYS}天，不参与拟合或评价",
            "scoring_holdout": f"> {TEST_END}，仅用于后续当前风险评分",
        },
        "summary": summary,
        "model_feature_count": len(MODEL_FEATURE_FIELDS),
        "model_features": MODEL_FEATURE_FIELDS,
        "identifier_fields": ["order_id", "customer_id", "order_date"],
        "forbidden_source_fields": sorted(FORBIDDEN_SOURCE_FIELDS),
        "forbidden_feature_intersection": forbidden_intersection,
        "leakage_check": "pass" if not forbidden_intersection else "fail",
        "credit_training_policy": "客户授信.csv仅提供当前状态，缺少历史版本；不进入历史训练特征，只用于当前风险展示和规则校验",
        "same_day_policy": "同一天其他订单、回款、应收和展期不进入当前订单历史特征",
        "outputs": {
            "labels": portable_path(processed / "order_labels.csv"),
            "features": portable_path(processed / "order_features.csv"),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="构建企业订单超期标签和无泄漏历史特征")
    parser.add_argument("source", type=Path)
    parser.add_argument("processed", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = build_training_data(args.source, args.processed, args.report)
    summary = report["summary"]
    eligible = sum(int(item["eligible"]) for item in summary["splits"].values())
    print(f"训练数据构建完成：{summary['total_orders']:,}个订单，{eligible:,}个可标注样本。")


if __name__ == "__main__":
    main()
