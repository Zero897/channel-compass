from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


FEATURES = ["order_amount", "prior_late_rate", "prior_sales_growth", "prior_order_count"]


def _generate_transactions(seed: int = 20260811) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    months = pd.date_range("2025-01-01", periods=18, freq="MS")
    for customer_index in range(12):
        base_risk = 0.03 + customer_index * 0.025
        base_sales = 80_000 + customer_index * 12_000
        for month_index, month in enumerate(months):
            deterioration = max(month_index - 10, 0) * (customer_index >= 8) * 0.035
            late_probability = min(base_risk + deterioration, 0.85)
            rows.append(
                {
                    "order_id": f"D{customer_index:02d}-{month_index:02d}",
                    "customer_id": f"DEMO-C{customer_index + 1:02d}",
                    "order_date": month.date().isoformat(),
                    "order_amount": round(float(base_sales * rng.uniform(0.75, 1.25)), 2),
                    "late_within_120d": int(rng.random() < late_probability),
                    "data_source": "portable_synthetic_demo",
                }
            )
    return pd.DataFrame(rows).sort_values(["customer_id", "order_date"])


def _build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    groups = transactions.groupby("customer_id", group_keys=False)
    output = transactions.copy()
    output["prior_order_count"] = groups.cumcount()
    output["prior_late_rate"] = groups["late_within_120d"].transform(
        lambda values: values.shift().expanding().mean()
    ).fillna(0.0)
    prior_amount = groups["order_amount"].shift()
    prior_prior_amount = groups["order_amount"].shift(2)
    output["prior_sales_growth"] = (
        (prior_amount - prior_prior_amount) / prior_prior_amount.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return output


def run_demo(output_dir: Path, seed: int = 20260811) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _generate_transactions(seed)
    features = _build_features(raw)
    split_date = "2026-01-01"
    train = features[features["order_date"] < split_date].copy()
    test = features[features["order_date"] >= split_date].copy()
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train[FEATURES])
    test_x = scaler.transform(test[FEATURES])
    model = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)
    model.fit(train_x, train["late_within_120d"])
    train_probability = model.predict_proba(train_x)[:, 1]
    test_probability = model.predict_proba(test_x)[:, 1]
    threshold = float(np.quantile(train_probability, 0.80))
    test["risk_probability"] = test_probability
    test["risk_score"] = test["risk_probability"].rank(pct=True, method="average") * 100
    test["risk_level"] = np.where(test["risk_probability"] >= threshold, "红色", "绿色")
    events = test[test["risk_level"] == "红色"].copy()
    events["risk_event_id"] = events["order_id"].map(lambda value: f"DEMO-R-{value}")
    events["claim_boundary"] = "风险分用于排序；合成数据结果不代表企业真实效果"

    raw.to_csv(output_dir / "raw_transactions.csv", index=False, encoding="utf-8-sig")
    features.to_csv(output_dir / "model_features.csv", index=False, encoding="utf-8-sig")
    events[
        [
            "risk_event_id",
            "customer_id",
            "order_id",
            "order_date",
            "order_amount",
            "risk_score",
            "risk_level",
            "claim_boundary",
        ]
    ].to_csv(output_dir / "dashboard_risk_events.csv", index=False, encoding="utf-8-sig")
    metrics = {
        "status": "pass",
        "data_source": "portable_synthetic_demo",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_pr_auc": round(float(average_precision_score(test["late_within_120d"], test_probability)), 6),
        "test_roc_auc": round(float(roc_auc_score(test["late_within_120d"], test_probability)), 6),
        "risk_events": int(len(events)),
        "threshold_policy": "训练期风险输出前20%对应阈值，固定后应用于未来测试期",
        "outputs": [
            "raw_transactions.csv",
            "model_features.csv",
            "dashboard_risk_events.csv",
        ],
    }
    (output_dir / "model_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="运行无需企业原始数据的渠智罗盘便携端到端演示")
    parser.add_argument("--output", type=Path, default=Path("demo_output"))
    args = parser.parse_args()
    result = run_demo(args.output)
    print(
        f"便携演示完成：{result['train_rows']}条训练、{result['test_rows']}条测试、"
        f"{result['risk_events']}条页面风险事件。"
    )


if __name__ == "__main__":
    main()
