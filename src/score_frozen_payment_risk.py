from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from build_company_training_data import MODEL_FEATURE_FIELDS
from project_paths import portable_path, sha256_file
from train_payment_risk_models import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    _apply_calibrator,
    _lightgbm_contributions,
    _load_frame,
    _logistic_contributions,
)


def _validate_bundle(bundle: Any) -> dict[str, Any]:
    if not isinstance(bundle, dict):
        raise ValueError("冻结模型文件必须包含模型字典")
    required = {"model_type", "probability_calibrator", "risk_threshold", "feature_names"}
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"冻结模型缺少字段：{', '.join(missing)}")
    feature_names = list(bundle["feature_names"])
    if feature_names != MODEL_FEATURE_FIELDS:
        raise ValueError("冻结模型特征契约与当前代码不一致，请重新核对模型版本")
    model_type = str(bundle["model_type"])
    if model_type == "logistic_regression" and "pipeline" not in bundle:
        raise ValueError("逻辑回归冻结模型缺少pipeline")
    if model_type == "lightgbm":
        lightgbm_required = {"model", "category_maps", "numeric_medians"}
        missing_lightgbm = sorted(lightgbm_required - set(bundle))
        if missing_lightgbm:
            raise ValueError(f"LightGBM冻结模型缺少字段：{', '.join(missing_lightgbm)}")
    if model_type not in {"logistic_regression", "lightgbm"}:
        raise ValueError(f"不支持的冻结模型类型：{model_type}")
    return bundle


def _prepare_lightgbm_holdout(frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    transformed = frame[MODEL_FEATURE_FIELDS].copy()
    category_maps = bundle["category_maps"]
    medians = bundle["numeric_medians"]
    for name in CATEGORICAL_FEATURES:
        transformed[name] = (
            transformed[name]
            .fillna("未知")
            .astype(str)
            .map(category_maps[name])
            .fillna(-1)
            .astype("int32")
        )
    for name in NUMERIC_FEATURES:
        transformed[name] = transformed[name].fillna(float(medians[name])).astype("float32")
    return transformed


def score_frozen_model(
    features_path: Path,
    model_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    if not features_path.exists():
        raise FileNotFoundError(f"缺少特征文件：{features_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"缺少冻结模型：{model_path}")
    bundle = _validate_bundle(joblib.load(model_path))
    frame = _load_frame(features_path)
    holdout = frame[
        (frame["dataset_split"] == "scoring_holdout")
        & (frame["label_status"] != "excluded_project_business")
    ].copy()
    if holdout.empty:
        raise ValueError("没有可供冻结模型评分的scoring_holdout订单")

    model_type = str(bundle["model_type"])
    if model_type == "logistic_regression":
        pipeline = bundle["pipeline"]
        raw_probability = pipeline.predict_proba(holdout[MODEL_FEATURE_FIELDS])[:, 1]
        contributions = _logistic_contributions(pipeline, holdout)
    else:
        transformed = _prepare_lightgbm_holdout(holdout, bundle)
        model = bundle["model"]
        raw_probability = model.predict_proba(transformed)[:, 1]
        contributions = _lightgbm_contributions(model, transformed, holdout)

    probability = _apply_calibrator(bundle["probability_calibrator"], raw_probability)
    threshold = float(bundle["risk_threshold"])
    output = holdout[
        ["order_id", "customer_id", "order_date", "label_status", "order_amount"]
    ].copy()
    output["risk_probability"] = probability
    output["model_top_contributions"] = contributions
    output["primary_model"] = model_type
    output["high_risk"] = (output["risk_probability"] >= threshold).astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    report: dict[str, Any] = {
        "status": "pass",
        "mode": "frozen_inference",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_type": model_type,
        "risk_threshold": threshold,
        "scored_rows": int(len(output)),
        "high_risk_rows": int(output["high_risk"].sum()),
        "features_path": portable_path(features_path),
        "features_sha256": sha256_file(features_path),
        "model_path": portable_path(model_path),
        "model_sha256": sha256_file(model_path),
        "output_path": portable_path(output_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="使用冻结模型生成当前订单风险评分")
    parser.add_argument("features", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = score_frozen_model(args.features, args.model, args.output, args.report)
    print(
        f"冻结模型推理完成：{report['scored_rows']}条订单，"
        f"{report['high_risk_rows']}条超过冻结阈值。"
    )


if __name__ == "__main__":
    main()
