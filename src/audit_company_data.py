from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from project_paths import portable_path
from typing import Iterable


csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


def _rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV没有表头：{path}")
        for row in reader:
            yield {key: (value or "").strip() for key, value in row.items()}


def _present(value: str) -> bool:
    return value not in {"", "无", "nan", "NaN", "None"}


def _identifier(value: str) -> str:
    """统一 CSV 中被导出为整数或 x.0 的业务标识。"""
    value = value.strip()
    if value.endswith(".0") and value[:-2].isdigit():
        return value[:-2]
    return value


def _number(value: str) -> float:
    if not _present(value):
        return 0.0
    try:
        return float(value.replace(",", ""))
    except ValueError as error:
        raise ValueError(f"无法解析数值：{value!r}") from error


def _day(value: str) -> str | None:
    if not _present(value):
        return None
    candidate = value[:10]
    try:
        date.fromisoformat(candidate)
    except ValueError as error:
        raise ValueError(f"无法解析日期：{value!r}") from error
    return candidate


def _update_range(stats: dict[str, object], value: str) -> None:
    parsed = _day(value)
    if not parsed:
        return
    current_min = stats.get("date_min")
    current_max = stats.get("date_max")
    stats["date_min"] = parsed if current_min is None else min(str(current_min), parsed)
    stats["date_max"] = parsed if current_max is None else max(str(current_max), parsed)


def _missing(row: dict[str, str], fields: list[str]) -> dict[str, int]:
    return {field: int(not _present(row.get(field, ""))) for field in fields}


def _add_missing(target: Counter[str], row: dict[str, str], fields: list[str]) -> None:
    target.update(_missing(row, fields))


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _coverage(source: set[str], reference: set[str]) -> dict[str, object]:
    matched = source & reference
    return {
        "source_distinct": len(source),
        "matched_distinct": len(matched),
        "coverage": _ratio(len(matched), len(source)),
        "unmatched_count": len(source - reference),
    }


def _table_base(file_name: str, grain: str) -> dict[str, object]:
    return {
        "file": file_name,
        "grain": grain,
        "row_count": 0,
        "date_min": None,
        "date_max": None,
    }


def audit(input_dir: Path, output_json: Path, output_markdown: Path) -> dict[str, object]:
    paths = {
        "sales": input_dir / "销售流水.csv",
        "payments": input_dir / "业务回款明细.csv",
        "contracts": input_dir / "增值合同签约明细.csv",
        "ar": input_dir / "应收快照_月末24期.csv",
        "inventory": input_dir / "库龄快照_季末8期.csv",
        "extensions": input_dir / "展期记录.csv",
        "credit": input_dir / "客户授信.csv",
    }
    missing_files = [path.name for path in paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"缺少企业数据文件：{missing_files}")

    tables: dict[str, dict[str, object]] = {}

    contract_stats = _table_base(paths["contracts"].name, "项目合同行")
    contract_missing: Counter[str] = Counter()
    project_contracts: set[str] = set()
    duplicate_contract_rows = 0
    contract_total_amount = 0.0
    contract_statuses: Counter[str] = Counter()
    for row in _rows(paths["contracts"]):
        contract_stats["row_count"] = int(contract_stats["row_count"]) + 1
        _update_range(contract_stats, row["申请日期"])
        _add_missing(contract_missing, row, ["申请日期", "合同编号", "客户名称", "销售金额"])
        contract_id = _identifier(row["合同编号"])
        if _present(contract_id):
            if contract_id in project_contracts:
                duplicate_contract_rows += 1
            project_contracts.add(contract_id)
        contract_total_amount += _number(row["销售金额"])
        contract_statuses[row["合同状态"] or "空"] += 1
    contract_stats.update(
        {
            "distinct_contracts": len(project_contracts),
            "duplicate_contract_rows": duplicate_contract_rows,
            "total_contract_amount": round(contract_total_amount, 2),
            "contract_status_distribution": dict(contract_statuses),
            "critical_missing": dict(contract_missing),
        }
    )
    tables["contracts"] = contract_stats

    sales_stats = _table_base(paths["sales"].name, "出库单行")
    sales_missing: Counter[str] = Counter()
    sales_customers: set[str] = set()
    sales_orders: set[str] = set()
    sales_contracts: set[str] = set()
    sales_materials: set[str] = set()
    sales_total_amount = 0.0
    sales_total_cost = 0.0
    negative_sales_rows = 0
    negative_sales_amount = 0.0
    project_contract_match_rows = 0
    project_id_present_rows = 0
    for row in _rows(paths["sales"]):
        sales_stats["row_count"] = int(sales_stats["row_count"]) + 1
        _update_range(sales_stats, row["出库日期"])
        _add_missing(
            sales_missing,
            row,
            ["出库日期", "客户编号", "销售订单号", "物料编码", "销售金额_折扣后_含税"],
        )
        customer = _identifier(row["客户编号"])
        order = _identifier(row["销售订单号"])
        contract = _identifier(row["合同号"])
        material = _identifier(row["物料编码"])
        if _present(customer):
            sales_customers.add(customer)
        if _present(order):
            sales_orders.add(order)
        if _present(contract):
            sales_contracts.add(contract)
        if _present(material):
            sales_materials.add(material)
        amount = _number(row["销售金额_折扣后_含税"])
        sales_total_amount += amount
        sales_total_cost += _number(row["出库成本金额"])
        if amount < 0 or _number(row["数量"]) < 0:
            negative_sales_rows += 1
            negative_sales_amount += amount
        if contract in project_contracts:
            project_contract_match_rows += 1
        if _present(row["项目编号"]):
            project_id_present_rows += 1
    sales_stats.update(
        {
            "distinct_customers": len(sales_customers),
            "distinct_orders": len(sales_orders),
            "distinct_contracts": len(sales_contracts),
            "distinct_materials": len(sales_materials),
            "total_sales_amount": round(sales_total_amount, 2),
            "total_cost_amount": round(sales_total_cost, 2),
            "gross_margin_ratio": round(
                (sales_total_amount - sales_total_cost) / sales_total_amount, 6
            )
            if sales_total_amount
            else 0.0,
            "negative_or_return_rows": negative_sales_rows,
            "negative_sales_amount": round(negative_sales_amount, 2),
            "project_contract_match_rows": project_contract_match_rows,
            "project_id_present_rows": project_id_present_rows,
            "critical_missing": dict(sales_missing),
        }
    )
    tables["sales"] = sales_stats

    payment_stats = _table_base(paths["payments"].name, "发票级回款")
    payment_missing: Counter[str] = Counter()
    payment_customers: set[str] = set()
    payment_orders: set[str] = set()
    payment_contracts: set[str] = set()
    payment_receipts: set[str] = set()
    payment_total = 0.0
    overdue_payment_total = 0.0
    overdue_interest_total = 0.0
    overdue_payment_rows = 0
    payment_project_rows = 0
    for row in _rows(paths["payments"]):
        payment_stats["row_count"] = int(payment_stats["row_count"]) + 1
        _update_range(payment_stats, row["回款日期"])
        _add_missing(
            payment_missing,
            row,
            ["回款日期", "客户编号", "销售订单号", "收款编号", "回款金额", "是否超期"],
        )
        customer = _identifier(row["客户编号"])
        order = _identifier(row["销售订单号"])
        contract = _identifier(row["合同号"])
        if _present(customer):
            payment_customers.add(customer)
        if _present(order):
            payment_orders.add(order)
        if _present(contract):
            payment_contracts.add(contract)
        if _present(row["收款编号"]):
            payment_receipts.add(row["收款编号"])
        amount = _number(row["回款金额"])
        payment_total += amount
        overdue_interest_total += _number(row["超期利息金额"])
        if row["是否超期"].upper() == "Y":
            overdue_payment_rows += 1
            overdue_payment_total += amount
        if contract in project_contracts:
            payment_project_rows += 1
    payment_stats.update(
        {
            "distinct_customers": len(payment_customers),
            "distinct_orders": len(payment_orders),
            "distinct_receipts": len(payment_receipts),
            "total_payment_amount": round(payment_total, 2),
            "overdue_payment_rows": overdue_payment_rows,
            "overdue_payment_row_ratio": _ratio(overdue_payment_rows, int(payment_stats["row_count"])),
            "overdue_payment_amount": round(overdue_payment_total, 2),
            "overdue_interest_amount": round(overdue_interest_total, 2),
            "project_contract_match_rows": payment_project_rows,
            "critical_missing": dict(payment_missing),
        }
    )
    tables["payments"] = payment_stats

    ar_stats = _table_base(paths["ar"].name, "月末×客户×合同×订单×物料")
    ar_missing: Counter[str] = Counter()
    ar_customers: set[str] = set()
    ar_orders: set[str] = set()
    ar_contracts: set[str] = set()
    ar_snapshot_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"receivable": 0.0, "overdue": 0.0, "overdue_30": 0.0, "overdue_60": 0.0, "rows": 0.0}
    )
    ar_project_rows = 0
    ar_extension_rows = 0
    for row in _rows(paths["ar"]):
        ar_stats["row_count"] = int(ar_stats["row_count"]) + 1
        snapshot = _day(row["快照时间"])
        if snapshot:
            _update_range(ar_stats, snapshot)
        _add_missing(
            ar_missing,
            row,
            ["快照时间", "客户编号", "销售订单号", "应收金额", "超期应收金额", "是否超期"],
        )
        customer = _identifier(row["客户编号"])
        order = _identifier(row["销售订单号"])
        contract = _identifier(row["合同号"])
        if _present(customer):
            ar_customers.add(customer)
        if _present(order):
            ar_orders.add(order)
        if _present(contract):
            ar_contracts.add(contract)
        if snapshot:
            bucket = ar_snapshot_totals[snapshot]
            bucket["rows"] += 1
            bucket["receivable"] += _number(row["应收金额"])
            bucket["overdue"] += _number(row["超期应收金额"])
            bucket["overdue_30"] += _number(row["超期30天以上金额"])
            bucket["overdue_60"] += _number(row["超期60天以上金额"])
        if contract in project_contracts:
            ar_project_rows += 1
        if row["是否展期"].upper() == "Y":
            ar_extension_rows += 1
    latest_snapshot = max(ar_snapshot_totals)
    latest_ar = ar_snapshot_totals[latest_snapshot]
    ar_stats.update(
        {
            "distinct_snapshots": len(ar_snapshot_totals),
            "distinct_customers": len(ar_customers),
            "distinct_orders": len(ar_orders),
            "project_contract_match_rows": ar_project_rows,
            "extension_rows": ar_extension_rows,
            "latest_snapshot": latest_snapshot,
            "latest_receivable_amount": round(latest_ar["receivable"], 2),
            "latest_overdue_amount": round(latest_ar["overdue"], 2),
            "latest_overdue_ratio": round(
                latest_ar["overdue"] / latest_ar["receivable"], 6
            )
            if latest_ar["receivable"]
            else 0.0,
            "latest_overdue_30_amount": round(latest_ar["overdue_30"], 2),
            "latest_overdue_60_amount": round(latest_ar["overdue_60"], 2),
            "snapshot_totals": {
                snapshot: {key: round(value, 2) for key, value in values.items()}
                for snapshot, values in sorted(ar_snapshot_totals.items())
            },
            "critical_missing": dict(ar_missing),
        }
    )
    tables["ar_snapshots"] = ar_stats

    inventory_stats = _table_base(paths["inventory"].name, "季末×物料×批次×仓库")
    inventory_missing: Counter[str] = Counter()
    inventory_materials: set[str] = set()
    inventory_snapshot_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"inventory": 0.0, "aged_180": 0.0, "aged_365": 0.0, "rows": 0.0}
    )
    borrowed_overdue_rows = 0
    for row in _rows(paths["inventory"]):
        inventory_stats["row_count"] = int(inventory_stats["row_count"]) + 1
        snapshot = _day(row["快照日期"])
        if snapshot:
            _update_range(inventory_stats, snapshot)
        _add_missing(
            inventory_missing,
            row,
            ["快照日期", "物料编码", "库存组织名称", "数量", "库龄", "含税总价"],
        )
        material = _identifier(row["物料编码"])
        if _present(material):
            inventory_materials.add(material)
        if snapshot:
            amount = _number(row["含税总价"])
            age = _number(row["库龄"])
            bucket = inventory_snapshot_totals[snapshot]
            bucket["rows"] += 1
            bucket["inventory"] += amount
            if age > 180:
                bucket["aged_180"] += amount
            if age > 365:
                bucket["aged_365"] += amount
        if row["是否超期"].upper() == "Y":
            borrowed_overdue_rows += 1
    latest_inventory_snapshot = max(inventory_snapshot_totals)
    latest_inventory = inventory_snapshot_totals[latest_inventory_snapshot]
    inventory_stats.update(
        {
            "distinct_snapshots": len(inventory_snapshot_totals),
            "distinct_materials": len(inventory_materials),
            "borrowed_overdue_rows": borrowed_overdue_rows,
            "latest_snapshot": latest_inventory_snapshot,
            "latest_inventory_amount": round(latest_inventory["inventory"], 2),
            "latest_aged_180_amount": round(latest_inventory["aged_180"], 2),
            "latest_aged_365_amount": round(latest_inventory["aged_365"], 2),
            "latest_aged_180_ratio": round(
                latest_inventory["aged_180"] / latest_inventory["inventory"], 6
            )
            if latest_inventory["inventory"]
            else 0.0,
            "snapshot_totals": {
                snapshot: {key: round(value, 2) for key, value in values.items()}
                for snapshot, values in sorted(inventory_snapshot_totals.items())
            },
            "critical_missing": dict(inventory_missing),
        }
    )
    tables["inventory_snapshots"] = inventory_stats

    extension_stats = _table_base(paths["extensions"].name, "展期单明细")
    extension_missing: Counter[str] = Counter()
    extension_customers: set[str] = set()
    extension_orders: set[str] = set()
    extension_groups: set[str] = set()
    extension_amount = 0.0
    for row in _rows(paths["extensions"]):
        extension_stats["row_count"] = int(extension_stats["row_count"]) + 1
        _update_range(extension_stats, row["快照时间"])
        _add_missing(
            extension_missing,
            row,
            ["快照时间", "客户编号", "销售订单号", "最终承诺还款日期", "应收金额", "gkey"],
        )
        customer = _identifier(row["客户编号"])
        order = _identifier(row["销售订单号"])
        if _present(customer):
            extension_customers.add(customer)
        if _present(order):
            extension_orders.add(order)
        if _present(row["gkey"]):
            extension_groups.add(row["gkey"])
        extension_amount += _number(row["应收金额"])
    extension_stats.update(
        {
            "distinct_customers": len(extension_customers),
            "distinct_orders": len(extension_orders),
            "distinct_extension_groups": len(extension_groups),
            "total_extension_receivable": round(extension_amount, 2),
            "critical_missing": dict(extension_missing),
        }
    )
    tables["extensions"] = extension_stats

    credit_stats = _table_base(paths["credit"].name, "客户当前授信状态")
    credit_missing: Counter[str] = Counter()
    credit_customers: set[str] = set()
    list_status: Counter[str] = Counter()
    zero_credit_customers = 0
    total_credit = 0.0
    total_frozen = 0.0
    for row in _rows(paths["credit"]):
        credit_stats["row_count"] = int(credit_stats["row_count"]) + 1
        _add_missing(
            credit_missing,
            row,
            ["客户编号_中台", "授信额度", "授信状态", "黑白名单状态"],
        )
        customer = _identifier(row["客户编号_中台"])
        if _present(customer):
            credit_customers.add(customer)
        credit = _number(row["授信额度"])
        total_credit += credit
        total_frozen += _number(row["冻结金额"])
        if credit == 0:
            zero_credit_customers += 1
        list_status[row["黑白名单状态"] or "空"] += 1
    credit_stats.update(
        {
            "distinct_customers": len(credit_customers),
            "zero_credit_customers": zero_credit_customers,
            "total_credit_limit": round(total_credit, 2),
            "total_frozen_amount": round(total_frozen, 2),
            "list_status_distribution": dict(list_status),
            "critical_missing": dict(credit_missing),
        }
    )
    tables["credit"] = credit_stats

    associations = {
        "payment_customers_in_sales": _coverage(payment_customers, sales_customers),
        "ar_customers_in_sales": _coverage(ar_customers, sales_customers),
        "credit_customers_in_sales": _coverage(credit_customers, sales_customers),
        "extension_customers_in_sales": _coverage(extension_customers, sales_customers),
        "payment_orders_in_sales": _coverage(payment_orders, sales_orders),
        "ar_orders_in_sales": _coverage(ar_orders, sales_orders),
        "extension_orders_in_sales": _coverage(extension_orders, sales_orders),
        "extension_orders_in_ar": _coverage(extension_orders, ar_orders),
        "project_contracts_in_sales": _coverage(project_contracts, sales_contracts),
        "project_contracts_in_payments": _coverage(project_contracts, payment_contracts),
        "project_contracts_in_ar": _coverage(project_contracts, ar_contracts),
        "inventory_materials_in_sales": _coverage(inventory_materials, sales_materials),
    }

    report = {
        "audit_status": "complete",
        "source": "AFFT模拟数据集",
        "input_directory": portable_path(input_dir),
        "data_safety": "脱敏模拟数据；报告不输出客户名称或逐行明细",
        "tables": tables,
        "associations": associations,
        "interpretation_constraints": [
            "应收快照按月截面分析，不得跨24期求和",
            "库龄快照无客户维度，只用于物料和仓库库存健康",
            "非项目类合同号为空或无法匹配签约表是正常现象",
            "负销售金额可能是退货或冲销，不能直接视为数据错误",
            "绝对金额经整体缩放，不代表真实企业规模",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_markdown.write_text(_markdown(report), encoding="utf-8")
    return report


def _money(value: object) -> str:
    return f"¥{float(value):,.0f}"


def _percent(value: object) -> str:
    return f"{float(value):.1%}"


def _markdown(report: dict[str, object]) -> str:
    tables = report["tables"]
    associations = report["associations"]
    ar = tables["ar_snapshots"]
    inventory = tables["inventory_snapshots"]
    lines = [
        "# AFFT企业模拟数据只读审计报告",
        "",
        "> 本报告仅输出聚合结果，不包含客户名称或逐行明细。绝对金额经过缩放，不代表真实企业规模。",
        "",
        "## 1. 数据规模",
        "",
        "| 表 | 行数 | 日期范围 | 粒度 |",
        "|---|---:|---|---|",
    ]
    for key in ("sales", "payments", "contracts", "ar_snapshots", "inventory_snapshots", "extensions", "credit"):
        item = tables[key]
        date_range = (
            f"{item['date_min']} 至 {item['date_max']}"
            if item.get("date_min")
            else "当前状态"
        )
        lines.append(f"| {item['file']} | {int(item['row_count']):,} | {date_range} | {item['grain']} |")
    lines.extend(
        [
            "",
            "## 2. 核心结果",
            "",
            f"- 销售客户：{tables['sales']['distinct_customers']}家；销售订单：{int(tables['sales']['distinct_orders']):,}个。",
            f"- 最新应收快照：{ar['latest_snapshot']}，应收余额{_money(ar['latest_receivable_amount'])}，逾期占比{_percent(ar['latest_overdue_ratio'])}。",
            f"- 最新库龄快照：{inventory['latest_snapshot']}，库存金额{_money(inventory['latest_inventory_amount'])}，180天以上占比{_percent(inventory['latest_aged_180_ratio'])}。",
            f"- 发生超期的回款行占比：{_percent(tables['payments']['overdue_payment_row_ratio'])}。",
            f"- 授信客户：{tables['credit']['distinct_customers']}家；零授信客户：{tables['credit']['zero_credit_customers']}家。",
            "",
            "## 3. 关键关联覆盖率",
            "",
            "| 关联 | 覆盖率 | 未匹配数 |",
            "|---|---:|---:|",
        ]
    )
    for key, label in (
        ("payment_customers_in_sales", "回款客户→销售客户"),
        ("ar_customers_in_sales", "应收客户→销售客户"),
        ("credit_customers_in_sales", "授信客户→销售客户"),
        ("payment_orders_in_sales", "回款订单→销售订单"),
        ("ar_orders_in_sales", "应收订单→销售订单"),
        ("extension_orders_in_ar", "展期订单→应收订单"),
        ("project_contracts_in_sales", "项目合同→销售合同"),
        ("inventory_materials_in_sales", "库存物料→销售物料"),
    ):
        item = associations[key]
        lines.append(f"| {label} | {_percent(item['coverage'])} | {item['unmatched_count']:,} |")
    lines.extend(
        [
            "",
            "## 4. 数据质量与使用判断",
            "",
            f"- 回款表有{int(tables['payments']['critical_missing']['收款编号']):,}行缺少收款编号，不能把收款编号直接当作行级主键。",
            f"- 应收快照有{int(ar['critical_missing']['客户编号']):,}行缺少客户和销售订单号，建模前应隔离而不是补造标识。",
            f"- 授信表有{int(tables['credit']['critical_missing']['授信状态']):,}行授信状态为空，但授信额度和名单状态完整，可分别使用。",
            "- 展期订单能回到销售流水，但在应收快照中无直接命中；展期只作为历史行为特征，不据此覆盖当前应收状态。",
            "- 标识字段统一去除纯数字末尾的“.0”后再关联；原始文件保持不变。",
            "",
            "## 5. 建模边界与首个预测目标",
            "",
            "- 主模型：订单在出库时预测后续是否发生超期回款，标签来自回款表“是否超期”。",
            "- 特征仅使用出库时点及之前的销售、历史回款、历史应收、授信和展期统计；实际回款日、超期利息等结果字段禁止进入特征。",
            "- 评分结果按客户聚合，形成开放订单后续超期风险和证据；不再表述为未经定义的“未来30天风险”。",
            "- 标签固定观察出库后120天；训练截止2025-03，验证为2025-08至09，测试为2026-02至03，窗口之间保留至少120天隔离带。",
            "- 库存模块单独按物料/仓库做180天以上库龄监测，不能宣称为某个渠道客户的库存风险。",
            "- 项目业务通过签约合同号连接销售、回款和应收，作为第二阶段扩展，不与分销主模型混训。",
            "",
            "## 6. 口径约束",
            "",
            "- 应收快照按月末截面使用，不能将24个月余额直接求和。",
            "- 库龄快照没有客户字段，只能建立物料/仓库库存健康模块。",
            "- 非项目类不要求合同号匹配签约表；项目类以签约合同号匹配为准。",
            "- 退货和冲销可能产生负销售额，需保留而不是删除。",
            "- 后续模型按观察时点切分，禁止使用未来回款和清账结果。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="流式审计AFFT企业模拟数据")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_markdown", type=Path)
    args = parser.parse_args()
    report = audit(args.input_dir, args.output_json, args.output_markdown)
    total_rows = sum(int(item["row_count"]) for item in report["tables"].values())
    print(f"企业数据审计完成：7张表，共{total_rows:,}行。")


if __name__ == "__main__":
    main()
