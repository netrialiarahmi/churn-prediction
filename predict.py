"""
predict.py — Load trained pipeline and generate submission.csv

Usage:
    python predict.py [--model models/pipeline.joblib] [--data data/test.csv]
"""

import os
import argparse
import logging
import warnings

import numpy as np
import pandas as pd
import joblib

from train import (
    clean_data,
    feature_engineering,
    encode_binary,
    encode_ordinal,
    encode_ohe,
    COLLAPSE_COLS,
    RESULTS_DIR,
    DATA_DIR,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_pipeline(model_path: str) -> dict:
    log.info("Loading pipeline from %s …", model_path)
    return joblib.load(model_path)


def preprocess_test(test: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Apply same preprocessing steps as training, using saved encoders."""
    binary_encoders = artifacts["binary_encoders"]
    ohe_cols = artifacts["ohe_cols"]
    feature_cols = artifacts["feature_cols"]

    # Feature engineering
    test = feature_engineering(test)

    # Encoding
    test = encode_ordinal(test)
    test, _ = encode_binary(test, fit_encoders=binary_encoders)
    test, _ = encode_ohe(test, fit_dummies=ohe_cols)

    # Align to training feature columns
    for col in feature_cols:
        if col not in test.columns:
            test[col] = 0

    return test[feature_cols]


def predict(pipeline: dict, X_test: pd.DataFrame) -> np.ndarray:
    """Generate stacked ensemble probabilities."""
    trained_models = pipeline["trained_models"]
    meta_lr = pipeline["meta_lr"]

    n = len(X_test)
    k = len(trained_models)
    test_preds = np.zeros((n, k))

    for model_idx, (name, fold_models) in enumerate(trained_models.items()):
        fold_test = np.stack(
            [m.predict_proba(X_test)[:, 1] for m in fold_models], axis=1
        )
        test_preds[:, model_idx] = fold_test.mean(axis=1)
        log.info("  Predictions from %s done", name)

    return meta_lr.predict_proba(test_preds)[:, 1]


def main(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load
    pipeline = load_pipeline(args.model)
    artifacts = pipeline["artifacts"]
    threshold = pipeline["threshold"]

    # Test data
    log.info("Loading test data from %s …", args.data)
    test = pd.read_csv(args.data)
    test_ids = test["id"].values

    # Clean + preprocess
    log.info("Preprocessing test data …")
    test_clean = clean_data(test)
    X_test = preprocess_test(test_clean, artifacts)

    # Predict
    log.info("Generating predictions (threshold=%.4f) …", threshold)
    proba = predict(pipeline, X_test)
    labels = (proba >= threshold).astype(int)

    # Save submission
    submission = pd.DataFrame({"id": test_ids, "Churn": labels})
    out_path = os.path.join(RESULTS_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)
    log.info("Submission saved → %s  (%d rows, churn rate=%.4f)",
             out_path, len(submission), labels.mean())

    # Verify against sample
    sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        assert list(submission.columns) == list(sample.columns), "Column mismatch!"
        assert len(submission) == len(sample), "Row count mismatch!"
        log.info("Format verified against sample_submission.csv ✓")

    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn prediction inference")
    parser.add_argument(
        "--model",
        default=os.path.join("models", "pipeline.joblib"),
        help="Path to saved pipeline.joblib",
    )
    parser.add_argument(
        "--data",
        default=os.path.join(DATA_DIR, "test.csv"),
        help="Path to test CSV",
    )
    args = parser.parse_args()
    main(args)
