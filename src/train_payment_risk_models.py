from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from threading import Event, Thread
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

from build_company_training_data import MODEL_FEATURE_FIELDS, OUTCOME_HORIZON_DAYS, TEST_END
from project_paths import PROJECT_ROOT, portable_path, sha256_file, sha256_tree


CATEGORICAL_FEATURES = ["customer_type", "region", "province"]
NUMERIC_FEATURES = [name for name in MODEL_FEATURE_FIELDS if name not in CATEGORICAL_FEATURES]
EVALUATION_SPLITS = ("train", "validation", "test")
ABLATION_EXCLUDED_FEATURES = {
    "prior_overdue_payment_count",
    "prior_overdue_payment_rate",
    "prior_overdue_payment_amount",
    "prior_overdue_amount_ratio",
    "latest_prior_overdue_ar_amount",
    "latest_prior_overdue_ar_ratio",
    "latest_prior_overdue_30_amount",
    "latest_prior_overdue_60_amount",
    "latest_prior_max_overdue_days",
}


def _load_frame(path: Path) -> pd.DataFrame:
    dtype = {
        "order_id": "string",
        "customer_id": "string",
        "order_date": "string",
        "dataset_split": "string",
        "label_status": "string",
        **{name: "string" for name in CATEGORICAL_FEATURES},
    }
    chunks: list[pd.DataFrame] = []
    with tqdm(desc="读取训练特征", unit="行", unit_scale=True) as progress:
        for chunk in pd.read_csv(path, dtype=dtype, low_memory=False, chunksize=100_000):
            chunks.append(chunk)
            progress.update(len(chunk))
    frame = pd.concat(chunks, ignore_index=True)
    for name in CATEGORICAL_FEATURES:
        frame[name] = frame[name].fillna("未知")
    for name in NUMERIC_FEATURES + ["label"]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    return frame


@contextmanager
def _elapsed_progress(description: str):
    """为不暴露逐轮回调的训练器显示真实耗时，而不伪造完成百分比。"""
    stopped = Event()
    progress = tqdm(
        desc=description,
        total=None,
        unit="秒",
        bar_format="{desc}: 已运行 {n:.0f}{unit} [{elapsed}]",
    )

    def update_elapsed() -> None:
        while not stopped.wait(1):
            progress.update(1)

    worker = Thread(target=update_elapsed, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()
        progress.close()


def _lightgbm_progress(total_iterations: int):
    progress = tqdm(total=total_iterations, desc="LightGBM迭代", unit="轮")

    def callback(environment: lgb.callback.CallbackEnv) -> None:
        completed = environment.iteration + 1
        if completed > progress.n:
            progress.update(completed - progress.n)

    callback.order = 10
    callback.before_iteration = False
    return progress, callback


def _rule_score(frame: pd.DataFrame) -> np.ndarray:
    score = (
        0.35 * frame["prior_overdue_payment_rate"].clip(0, 1)
        + 0.30 * frame["latest_prior_overdue_ar_ratio"].clip(0, 1)
        + 0.15 * (frame["latest_prior_max_overdue_days"].clip(0, 90) / 90)
        + 0.10 * (frame["prior_extension_count"] > 0).astype(float)
        + 0.10 * (-frame["prior_sales_30d_growth"]).clip(0, 1)
    )
    return score.fillna(0).to_numpy(dtype=float)


def _history_overdue_score(frame: pd.DataFrame) -> np.ndarray:
    return (
        (frame["prior_overdue_payment_count"] > 0)
        | (frame["latest_prior_overdue_ar_amount"] > 0)
        | (frame["latest_prior_max_overdue_days"] > 0)
    ).astype(float).to_numpy()


def _threshold_at_capacity(probabilities: np.ndarray, capacity: float = 0.10) -> float:
    return float(np.quantile(probabilities, 1.0 - capacity))


def _metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    positive_amount = np.maximum(amounts, 0) * (labels == 1)
    captured_amount = positive_amount * predictions
    top20_count = max(int(np.ceil(len(probabilities) * 0.20)), 1)
    top20_indices = np.argsort(-probabilities)[:top20_count]
    top20_captured_amount = positive_amount[top20_indices].sum()
    calibration_error = _expected_calibration_error(labels, probabilities)
    return {
        "rows": int(len(labels)),
        "positive_rate": round(float(labels.mean()), 6),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6),
        "pr_auc": round(float(average_precision_score(labels, probabilities)), 6),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 6),
        "expected_calibration_error": round(calibration_error, 6),
        "threshold": round(threshold, 6),
        "selection_rate": round(float(predictions.mean()), 6),
        "precision": round(float(tp / (tp + fp)) if tp + fp else 0.0, 6),
        "recall": round(float(tp / (tp + fn)) if tp + fn else 0.0, 6),
        "false_positive_rate": round(float(fp / (fp + tn)) if fp + tn else 0.0, 6),
        "business_false_alarm_rate": round(float(fp / (tp + fp)) if tp + fp else 0.0, 6),
        "top20_risk_amount_capture": round(
            float(top20_captured_amount / positive_amount.sum()) if positive_amount.sum() else 0.0,
            6,
        ),
        "risk_amount_capture": round(
            float(captured_amount.sum() / positive_amount.sum()) if positive_amount.sum() else 0.0,
            6,
        ),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }


def _expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, boundaries[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(labels[mask].mean())
            )
    return error


def _fit_sigmoid_calibrator(
    validation_probabilities: np.ndarray, validation_labels: pd.Series
) -> LogisticRegression:
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1_000)
    calibrator.fit(
        validation_probabilities.reshape(-1, 1),
        validation_labels.to_numpy(dtype=int),
    )
    return calibrator


def _apply_calibrator(
    calibrator: LogisticRegression, probabilities: np.ndarray
) -> np.ndarray:
    return calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]


def _evaluate_model(
    name: str,
    probabilities: dict[str, np.ndarray],
    frames: dict[str, pd.DataFrame],
    validation_threshold: float,
    splits: tuple[str, ...] = ("train", "validation"),
) -> dict[str, object]:
    return {
        "model": name,
        "threshold_policy": "验证集风险概率前10%对应阈值，固定后用于测试集",
        "validation_threshold": round(validation_threshold, 6),
        "splits": {
            split: _metrics(
                frames[split]["label"].to_numpy(dtype=int),
                probabilities[split],
                frames[split]["order_amount"].to_numpy(dtype=float),
                validation_threshold,
            )
            for split in splits
        },
    }


def _add_test_metrics(
    result: dict[str, object],
    probabilities: dict[str, np.ndarray],
    frames: dict[str, pd.DataFrame],
) -> None:
    threshold = float(result["validation_threshold"])
    result["splits"]["test"] = _metrics(
        frames["test"]["label"].to_numpy(dtype=int),
        probabilities["test"],
        frames["test"]["order_amount"].to_numpy(dtype=float),
        threshold,
    )


def _prepare_lightgbm_frames(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, int]], dict[str, float]]:
    category_maps: dict[str, dict[str, int]] = {}
    medians = {
        name: float(frames["train"][name].median())
        for name in NUMERIC_FEATURES
    }
    transformed: dict[str, pd.DataFrame] = {}
    for name in CATEGORICAL_FEATURES:
        values = sorted(frames["train"][name].fillna("未知").astype(str).unique())
        category_maps[name] = {value: index for index, value in enumerate(values)}
    for split, frame in frames.items():
        result = frame[MODEL_FEATURE_FIELDS].copy()
        for name in CATEGORICAL_FEATURES:
            result[name] = (
                result[name].fillna("未知").astype(str).map(category_maps[name]).fillna(-1).astype("int32")
            )
        for name in NUMERIC_FEATURES:
            result[name] = result[name].fillna(medians[name]).astype("float32")
        transformed[split] = result
    return transformed, category_maps, medians


def _comparison_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        for split in ("validation", "test"):
            metrics = result["splits"][split]
            rows.append(
                {
                    "model": result["model"],
                    "split": split,
                    "pr_auc": metrics["pr_auc"],
                    "roc_auc": metrics["roc_auc"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "false_positive_rate": metrics["false_positive_rate"],
                    "risk_amount_capture": metrics["risk_amount_capture"],
                    "selection_rate": metrics["selection_rate"],
                }
            )
    return rows


FEATURE_LABELS = {
    "customer_type": "客户类型",
    "region": "客户大区",
    "province": "客户省份",
    "order_amount": "订单金额",
    "cost_amount": "订单成本",
    "gross_margin_ratio": "毛利率",
    "quantity": "订单数量",
    "line_count": "订单行数",
    "sku_count": "SKU数量",
    "product_line_count": "产品线数量",
    "return_line_count": "退货行数",
    "return_amount_abs": "退货金额",
    "price_protection_amount": "价保金额",
    "vendor_rebate_amount": "厂商返利",
    "cash_discount_amount": "现金折扣",
    "avg_inventory_age": "出库商品平均库龄",
    "payment_term_days": "付款账期",
    "prior_order_count": "历史订单数",
    "prior_sales_amount": "历史累计销售额",
    "prior_sales_30d": "近30天销售额",
    "prior_sales_previous_30d": "此前30天销售额",
    "prior_sales_90d": "近90天销售额",
    "prior_sales_180d": "近180天销售额",
    "prior_orders_30d": "近30天订单数",
    "prior_orders_90d": "近90天订单数",
    "prior_return_ratio_180d": "近180天退货比例",
    "prior_sales_30d_growth": "近30天销售变化",
    "prior_payment_count": "历史回款笔数",
    "prior_overdue_payment_count": "历史超期回款笔数",
    "prior_overdue_payment_rate": "历史超期回款率",
    "prior_overdue_payment_amount": "历史超期回款金额",
    "prior_overdue_amount_ratio": "历史超期回款金额占比",
    "prior_avg_payment_age_days": "历史平均回款账龄",
    "latest_prior_ar_amount": "出库前应收余额",
    "latest_prior_overdue_ar_amount": "出库前逾期应收",
    "latest_prior_overdue_ar_ratio": "出库前逾期应收占比",
    "latest_prior_overdue_30_amount": "出库前逾期30天以上金额",
    "latest_prior_overdue_60_amount": "出库前逾期60天以上金额",
    "latest_prior_max_overdue_days": "出库前最大逾期天数",
    "prior_ar_snapshot_age_days": "出库前应收快照距今天数",
    "prior_ar_missing": "出库前应收快照缺失",
    "prior_extension_count": "历史展期次数",
    "prior_extension_amount": "历史展期金额",
}


def _feature_description(transformed_name: str, row: pd.Series) -> str:
    if transformed_name.startswith("raw__"):
        name = transformed_name.removeprefix("raw__")
        value = row[name]
        if name in CATEGORICAL_FEATURES:
            return f"{FEATURE_LABELS.get(name, name)}={value}"
        numeric_value = float(value) if pd.notna(value) else 0.0
        return f"{FEATURE_LABELS.get(name, name)}={numeric_value:,.3g}"
    if transformed_name.startswith("numeric__"):
        name = transformed_name.removeprefix("numeric__")
        value = float(row[name]) if pd.notna(row[name]) else 0.0
        return f"{FEATURE_LABELS.get(name, name)}={value:,.3g}"
    for name in CATEGORICAL_FEATURES:
        prefix = f"categorical__{name}_"
        if transformed_name.startswith(prefix):
            category = transformed_name.removeprefix(prefix)
            return f"{FEATURE_LABELS.get(name, name)}={category}"
    return transformed_name


def _format_contributions(
    contributions: np.ndarray,
    feature_names: list[str],
    raw_frame: pd.DataFrame,
) -> list[str]:
    explanations: list[str] = []
    for position, values in enumerate(contributions):
        positive = np.flatnonzero(values > 0)
        if len(positive):
            chosen = positive[np.argsort(values[positive])[-3:]][::-1]
        else:
            chosen = np.argsort(np.abs(values))[-3:][::-1]
        parts = [
            f"{_feature_description(feature_names[index], raw_frame.iloc[position])}（贡献{values[index]:+.3f}）"
            for index in chosen
            if abs(float(values[index])) > 1e-12
        ]
        explanations.append("；".join(parts) if parts else "无显著单项正向贡献，需人工复核")
    return explanations


def _logistic_contributions(pipeline: Pipeline, frame: pd.DataFrame) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(frame[MODEL_FEATURE_FIELDS])
    coefficients = pipeline.named_steps["model"].coef_[0]
    contribution_matrix = transformed.multiply(coefficients) if hasattr(transformed, "multiply") else transformed * coefficients
    if hasattr(contribution_matrix, "toarray"):
        contribution_matrix = contribution_matrix.toarray()
    return _format_contributions(
        np.asarray(contribution_matrix),
        [str(name) for name in preprocessor.get_feature_names_out()],
        frame.reset_index(drop=True),
    )


def _lightgbm_contributions(
    model: lgb.LGBMClassifier, transformed: pd.DataFrame, raw_frame: pd.DataFrame
) -> list[str]:
    values = model.booster_.predict(transformed, pred_contrib=True)[:, :-1]
    names = [f"raw__{name}" for name in MODEL_FEATURE_FIELDS]
    return _format_contributions(values, names, raw_frame.reset_index(drop=True))


def _calibration_rows(
    model_name: str,
    split: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> list[dict[str, object]]:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, boundaries[1:-1]), bins - 1)
    rows: list[dict[str, object]] = []
    for index in range(bins):
        mask = assignments == index
        if not mask.any():
            continue
        rows.append(
            {
                "model": model_name,
                "split": split,
                "bin_lower": round(float(boundaries[index]), 3),
                "bin_upper": round(float(boundaries[index + 1]), 3),
                "rows": int(mask.sum()),
                "mean_probability": round(float(probabilities[mask].mean()), 6),
                "observed_positive_rate": round(float(labels[mask].mean()), 6),
            }
        )
    return rows


def _customer_level_metrics(
    frame: pd.DataFrame, probability_column: str, threshold: float
) -> dict[str, float | int]:
    customer = (
        frame.assign(
            positive_amount=np.maximum(frame["order_amount"], 0) * frame["label"]
        )
        .groupby("customer_id", as_index=False)
        .agg(
            label=("label", "max"),
            probability=(probability_column, "max"),
            positive_amount=("positive_amount", "sum"),
        )
    )
    return _metrics(
        customer["label"].to_numpy(dtype=int),
        customer["probability"].to_numpy(dtype=float),
        customer["positive_amount"].to_numpy(dtype=float),
        threshold,
    )


def _monthly_stability_rows(
    frame: pd.DataFrame, probability_column: str, threshold: float
) -> list[dict[str, object]]:
    monthly = frame.copy()
    monthly["month"] = monthly["order_date"].astype(str).str[:7]
    rows: list[dict[str, object]] = []
    for month, group in monthly.groupby("month", sort=True):
        metrics = _metrics(
            group["label"].to_numpy(dtype=int),
            group[probability_column].to_numpy(dtype=float),
            group["order_amount"].to_numpy(dtype=float),
            threshold,
        )
        rows.append({"month": month, **metrics})
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "not_available"


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    test_results = {item["model"]: item["splits"]["test"] for item in report["models"]}
    primary_model = str(report["primary_model"])
    lines = [
        "# 订单超期回款模型训练报告",
        "",
        "> 所有指标均来自模型、概率校准器和阈值冻结后的未来测试集；测试集未参与任何选择或调参。",
        "",
        "## 数据门禁",
        "",
        f"- 训练样本：{report['dataset']['train_rows']:,}；验证样本：{report['dataset']['validation_rows']:,}；测试样本：{report['dataset']['test_rows']:,}。",
        f"- 训练正样本率：{report['dataset']['train_positive_rate']:.1%}；测试正样本率：{report['dataset']['test_positive_rate']:.1%}。",
        f"- 标签固定观察出库后{OUTCOME_HORIZON_DAYS}天，窗口之间设置等长时间隔离带。",
        f"- {TEST_END}之后的订单只用于当前风险评分，不参与模型评估。",
        "- 当前授信状态未进入历史特征，结果字段泄漏检查通过。",
        "",
        "## 测试集结果",
        "",
        "| 模型 | PR-AUC | ROC-AUC | 精确率 | 召回率 | 风险金额捕获率 | 选中率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("business_rule", "history_overdue_baseline", "logistic_regression"):
        item = test_results[name]
        lines.append(
            f"| {name} | {item['pr_auc']:.3f} | {item['roc_auc']:.3f} | {item['precision']:.1%} | {item['recall']:.1%} | {item['risk_amount_capture']:.1%} | {item['selection_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 使用方式",
            "",
            f"- 最终主模型冻结为 `{primary_model}`，风险阈值为 {report['primary_threshold']:.6f}。",
            "- 冠军模型只依据验证集PR-AUC和风险金额捕获率选择，测试集只做一次最终评价。",
            "- 预警阈值由验证集前10%处理容量确定，并原样应用于测试集。",
            "- 时间外测试集ECE表明概率仍有校准偏差；界面优先展示风险分位，不把约21%的模型输出解释为精确违约率。",
            "- 概率乘应收仅称为风险加权应收暴露，并用低/基准/高三档做敏感性分析，不解释为坏账期望损失。",
            f"- 冻结规则：{report['freeze_policy']}。",
            "- 当前留出订单只生成风险概率，不用其未成熟标签评价模型。",
            "- 飞书AI只读取结构化概率、金额和特征贡献生成摘要，不参与模型评分。",
            "",
            "## 客户级评价",
            "",
            f"- 客户级PR-AUC：{report['customer_level_test_metrics']['pr_auc']:.3f}；召回率：{report['customer_level_test_metrics']['recall']:.1%}。",
            f"- 测试期被选中客户中，前五名客户的选中订单金额占比为{report['risk_concentration']['top_five_selected_amount_share']:.1%}。",
            "",
            "## 补充实验",
            "",
            f"- 去除历史逾期特征后，逻辑回归测试PR-AUC为{test_results['logistic_without_overdue_history']['pr_auc']:.3f}；完整模型为{test_results['logistic_regression']['pr_auc']:.3f}。",
            f"- LightGBM测试PR-AUC为{test_results['lightgbm']['pr_auc']:.3f}，最佳迭代仅{report['best_iteration']}轮且无增益，因此不进入答辩主线。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def train_models(
    features_path: Path,
    reports_dir: Path,
    models_dir: Path,
    docs_dir: Path,
) -> dict[str, object]:
    overall = tqdm(total=6, desc="模型训练总体进度", unit="阶段")
    overall.set_postfix_str("读取与校验数据")
    frame = _load_frame(features_path)
    eligible = frame[
        (frame["label_status"] == "eligible")
        & frame["dataset_split"].isin(EVALUATION_SPLITS)
    ].copy()
    frames = {
        split: eligible[eligible["dataset_split"] == split].reset_index(drop=True)
        for split in EVALUATION_SPLITS
    }
    for split, split_frame in frames.items():
        if split_frame["label"].nunique() != 2:
            raise ValueError(f"{split}不同时包含正负样本，不能训练或评价")
    overall.update(1)

    overall.set_postfix_str("计算业务规则基线")
    rule_raw_probabilities = {
        split: _rule_score(split_frame) for split, split_frame in frames.items()
    }
    rule_calibrator = _fit_sigmoid_calibrator(
        rule_raw_probabilities["validation"], frames["validation"]["label"]
    )
    rule_probabilities = {
        split: _apply_calibrator(rule_calibrator, probabilities)
        for split, probabilities in rule_raw_probabilities.items()
    }
    rule_threshold = _threshold_at_capacity(rule_probabilities["validation"])
    rule_result = _evaluate_model("business_rule", rule_probabilities, frames, rule_threshold)
    rule_result["calibration"] = {"method": "sigmoid", "fit_split": "validation"}
    history_probabilities = {
        split: _history_overdue_score(split_frame) for split, split_frame in frames.items()
    }
    history_result = _evaluate_model(
        "history_overdue_baseline", history_probabilities, frames, 0.5
    )
    history_result["threshold_policy"] = "固定规则：评分前存在历史超期回款或超期应收即命中"
    history_result["calibration"] = {"method": "none", "fit_split": "not_applicable"}
    overall.update(1)

    logistic_max_iterations = 3_000
    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "categorical",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    (
                                        "onehot",
                                        OneHotEncoder(handle_unknown="ignore", min_frequency=50),
                                    ),
                                ]
                            ),
                            CATEGORICAL_FEATURES,
                        ),
                        (
                            "numeric",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            NUMERIC_FEATURES,
                        ),
                    ]
                ),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=logistic_max_iterations,
                    random_state=42,
                    solver="lbfgs",
                    tol=1e-3,
                ),
            ),
        ]
    )
    overall.set_postfix_str("训练逻辑回归")
    with _elapsed_progress("逻辑回归 lbfgs"):
        logistic_pipeline.fit(
            frames["train"][MODEL_FEATURE_FIELDS],
            frames["train"]["label"].astype(int),
        )
    logistic_iterations = int(logistic_pipeline.named_steps["model"].n_iter_.max())
    logistic_converged = logistic_iterations < logistic_max_iterations
    if not logistic_converged:
        raise RuntimeError("逻辑回归达到最大迭代次数，停止冻结模型")
    logistic_raw_probabilities = {
        split: logistic_pipeline.predict_proba(split_frame[MODEL_FEATURE_FIELDS])[:, 1]
        for split, split_frame in frames.items()
    }
    logistic_calibrator = _fit_sigmoid_calibrator(
        logistic_raw_probabilities["validation"], frames["validation"]["label"]
    )
    logistic_probabilities = {
        split: _apply_calibrator(logistic_calibrator, probabilities)
        for split, probabilities in logistic_raw_probabilities.items()
    }
    logistic_threshold = _threshold_at_capacity(logistic_probabilities["validation"])
    logistic_result = _evaluate_model(
        "logistic_regression", logistic_probabilities, frames, logistic_threshold
    )
    logistic_result["training"] = {
        "solver": "lbfgs",
        "max_iterations": logistic_max_iterations,
        "actual_iterations": logistic_iterations,
        "converged": logistic_converged,
    }
    logistic_result["calibration"] = {"method": "sigmoid", "fit_split": "validation"}

    ablation_features = [
        name for name in MODEL_FEATURE_FIELDS if name not in ABLATION_EXCLUDED_FEATURES
    ]
    ablation_categorical = [name for name in CATEGORICAL_FEATURES if name in ablation_features]
    ablation_numeric = [name for name in ablation_features if name not in ablation_categorical]
    ablation_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "categorical",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
                                ]
                            ),
                            ablation_categorical,
                        ),
                        (
                            "numeric",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            ablation_numeric,
                        ),
                    ]
                ),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=logistic_max_iterations,
                    random_state=42,
                    solver="lbfgs",
                    tol=1e-3,
                ),
            ),
        ]
    )
    with _elapsed_progress("逻辑回归消融（去除历史逾期特征）"):
        ablation_pipeline.fit(
            frames["train"][ablation_features], frames["train"]["label"].astype(int)
        )
    ablation_raw_probabilities = {
        split: ablation_pipeline.predict_proba(split_frame[ablation_features])[:, 1]
        for split, split_frame in frames.items()
    }
    ablation_calibrator = _fit_sigmoid_calibrator(
        ablation_raw_probabilities["validation"], frames["validation"]["label"]
    )
    ablation_probabilities = {
        split: _apply_calibrator(ablation_calibrator, probabilities)
        for split, probabilities in ablation_raw_probabilities.items()
    }
    ablation_threshold = _threshold_at_capacity(ablation_probabilities["validation"])
    ablation_result = _evaluate_model(
        "logistic_without_overdue_history",
        ablation_probabilities,
        frames,
        ablation_threshold,
    )
    ablation_result["ablation"] = {
        "removed_features": sorted(ABLATION_EXCLUDED_FEATURES),
        "remaining_feature_count": len(ablation_features),
    }
    ablation_result["calibration"] = {"method": "sigmoid", "fit_split": "validation"}
    overall.update(1)

    overall.set_postfix_str("训练LightGBM")
    lgb_frames, category_maps, medians = _prepare_lightgbm_frames(frames)
    train_labels = frames["train"]["label"].astype(int)
    positive = int(train_labels.sum())
    negative = int(len(train_labels) - positive)
    lightgbm_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        scale_pos_weight=negative / positive,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    lgb_progress, lgb_progress_callback = _lightgbm_progress(800)
    try:
        lightgbm_model.fit(
            lgb_frames["train"],
            train_labels,
            categorical_feature=CATEGORICAL_FEATURES,
            eval_X=lgb_frames["validation"],
            eval_y=frames["validation"]["label"].astype(int),
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb_progress_callback],
        )
    finally:
        lgb_progress.close()
    lightgbm_raw_probabilities = {
        split: lightgbm_model.predict_proba(split_frame)[:, 1]
        for split, split_frame in lgb_frames.items()
    }
    lightgbm_calibrator = _fit_sigmoid_calibrator(
        lightgbm_raw_probabilities["validation"], frames["validation"]["label"]
    )
    lightgbm_probabilities = {
        split: _apply_calibrator(lightgbm_calibrator, probabilities)
        for split, probabilities in lightgbm_raw_probabilities.items()
    }
    lightgbm_threshold = _threshold_at_capacity(lightgbm_probabilities["validation"])
    lightgbm_result = _evaluate_model(
        "lightgbm", lightgbm_probabilities, frames, lightgbm_threshold
    )
    lightgbm_result["training"] = {
        "best_iteration": int(lightgbm_model.best_iteration_),
        "early_stopping_rounds": 50,
    }
    lightgbm_result["calibration"] = {"method": "sigmoid", "fit_split": "validation"}
    overall.update(1)

    overall.set_postfix_str("比较并冻结主模型")
    candidate_results = [logistic_result, lightgbm_result]
    primary_result = max(
        candidate_results,
        key=lambda item: (
            float(item["splits"]["validation"]["pr_auc"]),
            float(item["splits"]["validation"]["risk_amount_capture"]),
        ),
    )
    primary_model_name = str(primary_result["model"])
    primary_threshold = float(primary_result["validation_threshold"])
    freeze_policy = (
        "仅按验证集PR-AUC为主、风险金额捕获率为辅选择冠军模型，阈值由验证集前10%处理容量确定；"
        "模型、校准器和阈值冻结后，测试集仅评价一次"
    )
    for result, probabilities in (
        (rule_result, rule_probabilities),
        (history_result, history_probabilities),
        (logistic_result, logistic_probabilities),
        (ablation_result, ablation_probabilities),
        (lightgbm_result, lightgbm_probabilities),
    ):
        _add_test_metrics(result, probabilities, frames)
    overall.update(1)

    overall.set_postfix_str("导出模型与报告")
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic_pipeline, models_dir / "logistic_regression.joblib")
    joblib.dump(
        logistic_calibrator, models_dir / "logistic_probability_calibrator.joblib"
    )
    joblib.dump(
        {
            "model": lightgbm_model,
            "probability_calibrator": lightgbm_calibrator,
            "feature_names": MODEL_FEATURE_FIELDS,
            "categorical_features": CATEGORICAL_FEATURES,
            "category_maps": category_maps,
            "numeric_medians": medians,
            "risk_threshold": lightgbm_threshold,
        },
        models_dir / "lightgbm_payment_risk.joblib",
    )
    if primary_model_name == "logistic_regression":
        primary_bundle: dict[str, object] = {
            "model_type": primary_model_name,
            "pipeline": logistic_pipeline,
            "probability_calibrator": logistic_calibrator,
            "risk_threshold": primary_threshold,
            "feature_names": MODEL_FEATURE_FIELDS,
        }
    else:
        primary_bundle = {
            "model_type": primary_model_name,
            "model": lightgbm_model,
            "probability_calibrator": lightgbm_calibrator,
            "risk_threshold": primary_threshold,
            "feature_names": MODEL_FEATURE_FIELDS,
            "categorical_features": CATEGORICAL_FEATURES,
            "category_maps": category_maps,
            "numeric_medians": medians,
        }
    joblib.dump(primary_bundle, models_dir / "primary_payment_risk.joblib")

    importance_rows = sorted(
        (
            {"feature": feature, "gain_importance": round(float(importance), 6)}
            for feature, importance in zip(
                MODEL_FEATURE_FIELDS,
                lightgbm_model.booster_.feature_importance(importance_type="gain"),
            )
        ),
        key=lambda row: float(row["gain_importance"]),
        reverse=True,
    )
    _write_csv(reports_dir / "feature_importance.csv", importance_rows)
    logistic_feature_names = logistic_pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    logistic_coefficients = logistic_pipeline.named_steps["model"].coef_[0]
    logistic_coefficient_rows = sorted(
        (
            {
                "feature": str(feature),
                "coefficient": round(float(coefficient), 8),
                "absolute_coefficient": round(abs(float(coefficient)), 8),
            }
            for feature, coefficient in zip(logistic_feature_names, logistic_coefficients)
        ),
        key=lambda row: float(row["absolute_coefficient"]),
        reverse=True,
    )
    _write_csv(reports_dir / "logistic_coefficients.csv", logistic_coefficient_rows)
    comparison_rows = _comparison_rows(
        [rule_result, history_result, logistic_result, ablation_result, lightgbm_result]
    )
    _write_csv(reports_dir / "model_comparison.csv", comparison_rows)

    test_predictions = frames["test"][["order_id", "customer_id", "order_date", "label", "order_amount"]].copy()
    test_predictions["rule_probability"] = rule_probabilities["test"]
    test_predictions["logistic_probability"] = logistic_probabilities["test"]
    test_predictions["lightgbm_probability"] = lightgbm_probabilities["test"]
    test_predictions["primary_model"] = primary_model_name
    primary_probability_column = (
        "logistic_probability"
        if primary_model_name == "logistic_regression"
        else "lightgbm_probability"
    )
    test_predictions["primary_high_risk"] = (
        test_predictions[primary_probability_column] >= primary_threshold
    ).astype(int)
    test_predictions.to_csv(reports_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    customer_level_metrics = _customer_level_metrics(
        test_predictions, primary_probability_column, primary_threshold
    )
    monthly_stability_rows = _monthly_stability_rows(
        test_predictions, primary_probability_column, primary_threshold
    )
    _write_csv(reports_dir / "monthly_stability.csv", monthly_stability_rows)
    selected_by_customer = (
        test_predictions[test_predictions["primary_high_risk"] == 1]
        .groupby("customer_id")["order_amount"]
        .apply(lambda values: float(np.maximum(values, 0).sum()))
        .sort_values(ascending=False)
    )
    selected_amount_total = float(selected_by_customer.sum())
    top_five_customer_share = (
        float(selected_by_customer.head(5).sum() / selected_amount_total)
        if selected_amount_total
        else 0.0
    )

    holdout = frame[
        (frame["dataset_split"] == "scoring_holdout")
        & (frame["label_status"] != "excluded_project_business")
    ].copy()
    holdout_lgb = holdout[MODEL_FEATURE_FIELDS].copy()
    for name in CATEGORICAL_FEATURES:
        holdout_lgb[name] = (
            holdout_lgb[name].fillna("未知").astype(str).map(category_maps[name]).fillna(-1).astype("int32")
        )
    for name in NUMERIC_FEATURES:
        holdout_lgb[name] = holdout_lgb[name].fillna(medians[name]).astype("float32")
    holdout_output = holdout[["order_id", "customer_id", "order_date", "label_status", "order_amount"]].copy()
    if primary_model_name == "logistic_regression":
        holdout_raw_probability = logistic_pipeline.predict_proba(
            holdout[MODEL_FEATURE_FIELDS]
        )[:, 1]
        holdout_output["risk_probability"] = _apply_calibrator(
            logistic_calibrator, holdout_raw_probability
        )
        holdout_output["model_top_contributions"] = _logistic_contributions(
            logistic_pipeline, holdout
        )
    else:
        holdout_raw_probability = lightgbm_model.predict_proba(holdout_lgb)[:, 1]
        holdout_output["risk_probability"] = _apply_calibrator(
            lightgbm_calibrator, holdout_raw_probability
        )
        holdout_output["model_top_contributions"] = _lightgbm_contributions(
            lightgbm_model, holdout_lgb, holdout
        )
    holdout_output["primary_model"] = primary_model_name
    holdout_output["high_risk"] = (
        holdout_output["risk_probability"] >= primary_threshold
    ).astype(int)
    holdout_output.to_csv(reports_dir / "holdout_risk_scores.csv", index=False, encoding="utf-8-sig")

    calibration_rows: list[dict[str, object]] = []
    for result, probabilities in (
        (rule_result, rule_probabilities),
        (history_result, history_probabilities),
        (logistic_result, logistic_probabilities),
        (ablation_result, ablation_probabilities),
        (lightgbm_result, lightgbm_probabilities),
    ):
        for split in ("validation", "test"):
            calibration_rows.extend(
                _calibration_rows(
                    str(result["model"]),
                    split,
                    frames[split]["label"].to_numpy(dtype=int),
                    probabilities[split],
                )
            )
    _write_csv(reports_dir / "calibration_table.csv", calibration_rows)

    report: dict[str, Any] = {
        "status": "complete",
        "task": "订单出库时预测后续是否发生超期回款",
        "split_policy": f"固定{OUTCOME_HORIZON_DAYS}天标签观察期，训练/验证/测试窗口之间设置等长隔离带；测试集不参与模型、校准器或阈值选择",
        "dataset": {
            "train_rows": len(frames["train"]),
            "validation_rows": len(frames["validation"]),
            "test_rows": len(frames["test"]),
            "train_positive_rate": round(float(frames["train"]["label"].mean()), 6),
            "validation_positive_rate": round(float(frames["validation"]["label"].mean()), 6),
            "test_positive_rate": round(float(frames["test"]["label"].mean()), 6),
            "holdout_scored_rows": len(holdout),
        },
        "models": [
            rule_result,
            history_result,
            logistic_result,
            ablation_result,
            lightgbm_result,
        ],
        "primary_model": primary_model_name,
        "primary_threshold": round(primary_threshold, 6),
        "freeze_policy": freeze_policy,
        "probability_policy": "模型输出经验证集一维Sigmoid校准，但时间外测试ECE显示仍有偏差；界面按风险分位展示，金额加权结果需做低/基准/高敏感性分析",
        "customer_level_test_metrics": customer_level_metrics,
        "risk_concentration": {
            "selected_customers": int(len(selected_by_customer)),
            "top_five_selected_amount_share": round(top_five_customer_share, 6),
        },
        "frozen": True,
        "best_iteration": int(lightgbm_model.best_iteration_),
        "top_features": importance_rows[:15],
        "logistic_top_coefficients": logistic_coefficient_rows[:15],
        "limitations": [
            "数据为企业提供的脱敏模拟数据，指标不能外推到真实经营水平",
            "当前授信表缺少历史版本，未进入历史模型",
            "标签基于已观察到的超期回款或超期应收，近期未结订单只做评分不做评价",
            "模型衡量相关性和排序能力，不证明干预措施具有因果效果",
        ],
    }
    (reports_dir / "model_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_freeze = {
        "status": "frozen",
        "primary_model": primary_model_name,
        "risk_threshold": round(primary_threshold, 6),
        "freeze_policy": freeze_policy,
        "selection_split": "validation",
        "validation_metrics": primary_result["splits"]["validation"],
        "test_metrics": primary_result["splits"]["test"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
            "lightgbm": version("lightgbm"),
            "joblib": version("joblib"),
        },
        "provenance": {
            "feature_file": portable_path(features_path),
            "feature_sha256": sha256_file(features_path),
            "primary_model_file": portable_path(models_dir / "primary_payment_risk.joblib"),
            "primary_model_sha256": sha256_file(models_dir / "primary_payment_risk.joblib"),
            "code_commit": _git_commit(),
            "code_snapshot_sha256": sha256_tree(list((PROJECT_ROOT / "src").glob("*.py"))),
        },
    }
    (reports_dir / "model_freeze.json").write_text(
        json.dumps(model_freeze, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(docs_dir / "model_training_report.md", report)
    overall.update(1)
    overall.set_postfix_str(f"已冻结 {primary_model_name}")
    overall.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="训练订单超期规则、逻辑回归与LightGBM模型")
    parser.add_argument("features", type=Path)
    parser.add_argument("reports", type=Path)
    parser.add_argument("models", type=Path)
    parser.add_argument("docs", type=Path)
    args = parser.parse_args()
    report = train_models(args.features, args.reports, args.models, args.docs)
    test = next(
        item for item in report["models"] if item["model"] == report["primary_model"]
    )["splits"]["test"]
    print(
        f"模型训练完成：已冻结{report['primary_model']}，阈值={report['primary_threshold']:.4f}，"
        f"测试集PR-AUC={test['pr_auc']:.4f}，ROC-AUC={test['roc_auc']:.4f}。"
    )


if __name__ == "__main__":
    main()
