from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from project_paths import portable_path


PRIOR_OVERDUE_FIELDS = [
    "prior_overdue_payment_count",
    "prior_overdue_payment_amount",
    "latest_prior_overdue_ar_amount",
    "latest_prior_overdue_30_amount",
    "latest_prior_overdue_60_amount",
    "latest_prior_max_overdue_days",
]


def analyze_early_warning(
    features_path: Path,
    labels_path: Path,
    predictions_path: Path,
    cases_path: Path,
    report_path: Path,
) -> dict[str, object]:
    features = pd.read_csv(
        features_path,
        usecols=["order_id", *PRIOR_OVERDUE_FIELDS],
        dtype={"order_id": "string"},
        low_memory=False,
    )
    labels = pd.read_csv(
        labels_path,
        usecols=["order_id", "customer_id", "label", "label_evidence_date"],
        dtype={"order_id": "string", "customer_id": "string"},
        low_memory=False,
    )
    predictions = pd.read_csv(
        predictions_path,
        dtype={"order_id": "string", "customer_id": "string"},
        low_memory=False,
    )
    first_evidence = (
        labels[labels["label"] == 1]
        .dropna(subset=["label_evidence_date"])
        .groupby("customer_id")["label_evidence_date"]
        .min()
    )
    candidates = (
        predictions[(predictions["primary_high_risk"] == 1) & (predictions["label"] == 1)]
        .merge(features, on="order_id", how="left")
        .merge(labels[["order_id", "label_evidence_date"]], on="order_id", how="left")
    )
    candidates["customer_first_overdue_evidence_date"] = candidates["customer_id"].map(
        first_evidence
    )
    no_prior_overdue = candidates[PRIOR_OVERDUE_FIELDS].fillna(0).abs().sum(axis=1) == 0
    score_date = pd.to_datetime(candidates["order_date"], errors="coerce")
    evidence_date = pd.to_datetime(
        candidates["customer_first_overdue_evidence_date"], errors="coerce"
    )
    strict = candidates[no_prior_overdue & evidence_date.gt(score_date)].copy()
    strict["lead_days"] = (
        pd.to_datetime(strict["customer_first_overdue_evidence_date"])
        - pd.to_datetime(strict["order_date"])
    ).dt.days
    columns = [
        "order_id",
        "customer_id",
        "order_date",
        "customer_first_overdue_evidence_date",
        "lead_days",
        "logistic_probability",
        "order_amount",
    ]
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    strict.reindex(columns=columns).to_csv(cases_path, index=False, encoding="utf-8-sig")
    report = {
        "status": "pass",
        "strict_customer_early_warning_cases": int(len(strict)),
        "strict_customer_early_warning_customers": int(strict["customer_id"].nunique()),
        "definition": "评分时客户无任何历史逾期特征，且客户首次观测到逾期证据的日期晚于评分日",
        "claim_policy": (
            "存在可复核案例，可按评分日、风险发生日和提前天数展示"
            if len(strict)
            else "未发现严格客户级提前预警案例；主展示必须称为存量风险向新增订单扩散监测"
        ),
        "cases_output": portable_path(cases_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="审计客户首次逾期前预警证据")
    parser.add_argument("features", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = analyze_early_warning(
        args.features, args.labels, args.predictions, args.cases, args.report
    )
    print(f"严格客户级提前预警案例：{report['strict_customer_early_warning_cases']}条")


if __name__ == "__main__":
    main()
