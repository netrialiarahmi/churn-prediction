"""
train.py — Churn Prediction Pipeline
Kaggle Playground Series S6E3

Phases:
  1. Load & clean data
  2. Feature engineering
  3. Preprocessing
  4. 5-fold Stratified CV with XGBoost, LightGBM, CatBoost, Random Forest
  5. Optuna hyperparameter tuning
  6. Stacking ensemble (LR meta-learner)
  7. SHAP analysis
  8. Save pipeline artifacts
"""

import os
import warnings
import argparse
import logging

import numpy as np
import pandas as pd
import joblib
import optuna
from scipy.stats import mstats
from scipy.optimize import minimize

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"
N_SPLITS = 5
RANDOM_STATE = 42
OPTUNA_TRIALS = 50  # reduce for faster runs; increase to 150-200 for best results

BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "Churn",
    # After collapsing "No internet service" / "No phone service" → "No",
    # these become binary Yes/No columns:
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
ORDINAL_COLS = {"Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2}}
OHE_COLS = ["InternetService", "PaymentMethod"]
COLLAPSE_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------
def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Loading data …")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    log.info("Train shape: %s | Test shape: %s", train.shape, test.shape)
    return train, test


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Replace blank TotalCharges strings with tenure × MonthlyCharges."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    mask = df["TotalCharges"].isna()
    if mask.sum() > 0:
        log.info("Imputing %d blank TotalCharges with tenure × MonthlyCharges", mask.sum())
        df.loc[mask, "TotalCharges"] = (
            df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"]
        )
    return df


def collapse_no_service(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse 'No internet service' / 'No phone service' → 'No'."""
    df = df.copy()
    for col in COLLAPSE_COLS:
        if col in df.columns:
            df[col] = df[col].replace(
                {"No internet service": "No", "No phone service": "No"}
            )
    return df


def winsorize_numerics(df: pd.DataFrame, limits: tuple = (0.01, 0.01)) -> pd.DataFrame:
    """Cap outliers at P1/P99 for numeric columns."""
    df = df.copy()
    for col in ["MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df[col] = mstats.winsorize(df[col].fillna(0), limits=limits).data
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = fix_total_charges(df)
    df = collapse_no_service(df)
    df = winsorize_numerics(df)
    return df


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------
def encode_binary(df: pd.DataFrame, fit_encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    encoders = fit_encoders or {}
    cols = [c for c in BINARY_COLS if c in df.columns]
    for col in cols:
        if col not in encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])
    return df, encoders


def encode_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in ORDINAL_COLS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    return df


def encode_ohe(
    df: pd.DataFrame, fit_dummies: list | None = None
) -> tuple[pd.DataFrame, list]:
    df = df.copy()
    cols = [c for c in OHE_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cols, drop_first=False)
    if fit_dummies is None:
        fit_dummies = [c for c in df.columns if any(c.startswith(oc + "_") for oc in cols)]
    else:
        # align columns
        for col in fit_dummies:
            if col not in df.columns:
                df[col] = 0
        df = df[[c for c in df.columns if c not in fit_dummies] + fit_dummies]
    return df, fit_dummies


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Domain features
    df["AvgMonthlyCharge"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["ServiceCount"] = (
        (df[COLLAPSE_COLS] == "Yes").sum(axis=1)
        if all(c in df.columns for c in COLLAPSE_COLS)
        else 0
    )
    df["TenureBin"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)
    df["IsFiberOptic"] = (df.get("InternetService", "") == "Fiber optic").astype(int)
    df["IsElectronicCheck"] = (df.get("PaymentMethod", "") == "Electronic check").astype(int)
    df["IsMonthToMonth"] = (df.get("Contract", "") == "Month-to-month").astype(int)
    df["NewCustomer"] = (df["tenure"] <= 6).astype(int)
    df["HighValueCustomer"] = (
        df["TotalCharges"] > df["TotalCharges"].quantile(0.75)
    ).astype(int)
    df["TenureChargeInteraction"] = df["tenure"] * df["MonthlyCharges"]
    df["ContractTenureInteraction"] = df.get("Contract", pd.Series(0, index=df.index)).replace(
        {"Month-to-month": 0, "One year": 1, "Two year": 2}
    ).fillna(0).astype(int) * df["tenure"]

    return df


# ---------------------------------------------------------------------------
# Full preprocessing pipeline (fit on train, apply to test)
# ---------------------------------------------------------------------------
def preprocess(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    """
    Returns:
        X_train, X_test  (feature matrices, no id/target)
        y_train          (numpy array)
        artifacts        (dict of encoders/columns for predict.py)
    """
    # Feature engineering BEFORE encoding (needs original string values)
    train = feature_engineering(train)
    test = feature_engineering(test)

    # Clean encode
    train = encode_ordinal(train)
    test = encode_ordinal(test)

    train, binary_encoders = encode_binary(train)
    test, _ = encode_binary(test, fit_encoders=binary_encoders)

    train, ohe_cols = encode_ohe(train)
    test, _ = encode_ohe(test, fit_dummies=ohe_cols)

    # Target
    y_train = train["Churn"].values

    # Drop columns we don't need
    drop_cols = ["id", "Churn"]
    X_train = train.drop(columns=[c for c in drop_cols if c in train.columns])
    X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])

    # Align columns
    shared = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[shared]
    X_test = X_test[shared]

    # Ensure no boolean columns (convert to int for model compatibility)
    bool_cols = X_train.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        X_train[bool_cols] = X_train[bool_cols].astype(int)
        X_test[bool_cols] = X_test[bool_cols].astype(int)

    log.info("X_train: %s | X_test: %s | y_train mean: %.4f",
             X_train.shape, X_test.shape, y_train.mean())

    artifacts = {
        "binary_encoders": binary_encoders,
        "ohe_cols": ohe_cols,
        "feature_cols": list(X_train.columns),
    }

    return X_train, X_test, y_train, artifacts


# ---------------------------------------------------------------------------
# Optuna objective factories
# ---------------------------------------------------------------------------
def xgb_objective(trial, X, y, cv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        "tree_method": "hist",
        "eval_metric": "auc",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def lgb_objective(trial, X, y, cv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "is_unbalance": True,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def cat_objective(trial, X, y, cv):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 800),
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "auto_class_weights": "Balanced",
        "random_seed": RANDOM_STATE,
        "verbose": 0,
    }
    model = CatBoostClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def rf_objective(trial, X, y, cv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
def tune_model(name: str, objective_fn, X, y, cv, n_trials: int = OPTUNA_TRIALS):
    log.info("Tuning %s (%d trials) …", name, n_trials)
    study = optuna.create_study(direction="maximize", study_name=name)
    study.optimize(
        lambda trial: objective_fn(trial, X, y, cv),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    log.info("%s best AUC: %.5f | params: %s", name, study.best_value, study.best_params)
    return study.best_params


# ---------------------------------------------------------------------------
# OOF predictions (stacking level-0)
# ---------------------------------------------------------------------------
def get_oof_predictions(
    models_params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> tuple[np.ndarray, dict]:
    """
    Returns:
        oof_preds: (n_samples, n_models) array of OOF probabilities
        trained_models: dict of list of fold-fitted models
    """
    n = len(X)
    k = len(models_params)
    oof_preds = np.zeros((n, k))
    trained_models = {name: [] for name in models_params}

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        log.info("  Fold %d/%d …", fold_idx + 1, cv.n_splits)

        for model_idx, (name, (ModelClass, params)) in enumerate(models_params.items()):
            model = ModelClass(**params)
            model.fit(X_tr, y_tr)
            oof_preds[val_idx, model_idx] = model.predict_proba(X_val)[:, 1]
            trained_models[name].append(model)

    for model_idx, name in enumerate(models_params):
        fold_auc = roc_auc_score(y, oof_preds[:, model_idx])
        log.info("  OOF AUC [%s]: %.5f", name, fold_auc)

    return oof_preds, trained_models


# ---------------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------------
def optimise_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    thresholds = np.linspace(0.1, 0.9, 161)
    best_acc, best_thr = 0.0, 0.5
    for thr in thresholds:
        acc = accuracy_score(y_true, (proba >= thr).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    log.info("Optimal threshold: %.4f → Accuracy: %.5f", best_thr, best_acc)
    return best_thr


# ---------------------------------------------------------------------------
# Test predictions (average across folds)
# ---------------------------------------------------------------------------
def predict_test(trained_models: dict, X_test: pd.DataFrame) -> np.ndarray:
    """Average fold predictions for each model."""
    n = len(X_test)
    k = len(trained_models)
    test_preds = np.zeros((n, k))
    for model_idx, (name, fold_models) in enumerate(trained_models.items()):
        fold_test = np.stack(
            [m.predict_proba(X_test)[:, 1] for m in fold_models], axis=1
        )
        test_preds[:, model_idx] = fold_test.mean(axis=1)
    return test_preds


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------
def shap_analysis(model, X_sample: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()
        log.info("SHAP summary plot saved → %s", output_path)
    except Exception as exc:
        log.warning("SHAP analysis failed: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    train_raw, test_raw = load_data(
        os.path.join(DATA_DIR, "train.csv"),
        os.path.join(DATA_DIR, "test.csv"),
    )

    # ── Clean ─────────────────────────────────────────────────────────────
    log.info("Cleaning data …")
    train_clean = clean_data(train_raw)
    test_clean = clean_data(test_raw)

    # ── Preprocess ────────────────────────────────────────────────────────
    X_train, X_test, y_train, artifacts = preprocess(train_clean, test_clean)
    test_ids = test_raw["id"].values

    # ── CV setup ──────────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # ── Baseline: Logistic Regression ─────────────────────────────────────
    log.info("Baseline: Logistic Regression …")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    )
    lr_auc = cross_val_score(lr, X_scaled, y_train, cv=cv, scoring="roc_auc").mean()
    log.info("Baseline LR AUC: %.5f", lr_auc)

    # ── Hyperparameter tuning ─────────────────────────────────────────────
    if args.tune:
        log.info("Hyperparameter tuning …")
        xgb_params = tune_model("XGBoost", xgb_objective, X_train, y_train, cv, args.trials)
        lgb_params = tune_model("LightGBM", lgb_objective, X_train, y_train, cv, args.trials)
        cat_params = tune_model("CatBoost", cat_objective, X_train, y_train, cv, args.trials)
        rf_params = tune_model("RandomForest", rf_objective, X_train, y_train, cv, args.trials)

        # Merge fixed params into tuned params
        xgb_params.update({"tree_method": "hist", "eval_metric": "auc",
                            "random_state": RANDOM_STATE, "n_jobs": -1})
        lgb_params.update({"is_unbalance": True, "random_state": RANDOM_STATE,
                            "n_jobs": -1, "verbose": -1})
        cat_params.update({"auto_class_weights": "Balanced",
                           "random_seed": RANDOM_STATE, "verbose": 0})
        rf_params.update({"class_weight": "balanced",
                          "random_state": RANDOM_STATE, "n_jobs": -1})
    else:
        log.info("Using default (non-tuned) model params …")
        xgb_params = {
            "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "scale_pos_weight": 2.7, "tree_method": "hist", "eval_metric": "auc",
            "random_state": RANDOM_STATE, "n_jobs": -1,
        }
        lgb_params = {
            "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
            "num_leaves": 63, "subsample": 0.8, "colsample_bytree": 0.8,
            "is_unbalance": True, "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
        }
        cat_params = {
            "iterations": 400, "depth": 5, "learning_rate": 0.05,
            "auto_class_weights": "Balanced", "random_seed": RANDOM_STATE, "verbose": 0,
        }
        rf_params = {
            "n_estimators": 200, "max_depth": 10, "min_samples_split": 5,
            "class_weight": "balanced", "random_state": RANDOM_STATE, "n_jobs": -1,
        }

    models_params = {
        "XGBoost": (xgb.XGBClassifier, xgb_params),
        "LightGBM": (lgb.LGBMClassifier, lgb_params),
        "CatBoost": (CatBoostClassifier, cat_params),
        "RandomForest": (RandomForestClassifier, rf_params),
    }

    # ── OOF predictions ───────────────────────────────────────────────────
    log.info("Generating OOF predictions …")
    oof_preds, trained_models = get_oof_predictions(models_params, X_train, y_train, cv)

    # ── Test predictions ──────────────────────────────────────────────────
    log.info("Generating test predictions …")
    test_preds = predict_test(trained_models, X_test)

    # ── Meta-learner (Stacking) ───────────────────────────────────────────
    log.info("Fitting stacking meta-learner …")
    meta_lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    meta_lr.fit(oof_preds, y_train)
    stack_proba = meta_lr.predict_proba(oof_preds)[:, 1]
    stack_auc = roc_auc_score(y_train, stack_proba)
    log.info("Stack OOF AUC: %.5f", stack_auc)

    # ── Threshold tuning ─────────────────────────────────────────────────
    optimal_thr = optimise_threshold(y_train, stack_proba)
    stack_preds = (stack_proba >= optimal_thr).astype(int)
    stack_acc = accuracy_score(y_train, stack_preds)
    stack_f1 = f1_score(y_train, stack_preds)
    cm = confusion_matrix(y_train, stack_preds)

    log.info("=== Final OOF Metrics ===")
    log.info("  AUC-ROC : %.5f", stack_auc)
    log.info("  Accuracy: %.5f", stack_acc)
    log.info("  F1      : %.5f", stack_f1)
    log.info("  Confusion Matrix:\n%s", cm)

    # ── Test submission ───────────────────────────────────────────────────
    test_stack_proba = meta_lr.predict_proba(test_preds)[:, 1]
    test_labels = (test_stack_proba >= optimal_thr).astype(int)

    submission = pd.DataFrame({"id": test_ids, "Churn": test_labels})
    submission_path = os.path.join(RESULTS_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    log.info("Submission saved → %s", submission_path)

    # Verify format matches sample
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    assert list(submission.columns) == list(sample.columns), "Column mismatch!"
    assert len(submission) == len(sample), "Row count mismatch!"
    log.info("Submission format verified ✓")

    # ── SHAP ──────────────────────────────────────────────────────────────
    log.info("Computing SHAP values …")
    # Use last fold XGBoost model
    xgb_model = trained_models["XGBoost"][-1]
    shap_sample = X_train.sample(min(2000, len(X_train)), random_state=RANDOM_STATE)
    shap_analysis(xgb_model, shap_sample, os.path.join(RESULTS_DIR, "shap_summary.png"))

    # ── Save artifacts ─────────────────────────────────────────────────────
    log.info("Saving model artifacts …")
    joblib.dump(
        {
            "trained_models": trained_models,
            "meta_lr": meta_lr,
            "threshold": optimal_thr,
            "artifacts": artifacts,
            "scaler": scaler,
        },
        os.path.join(MODELS_DIR, "pipeline.joblib"),
    )
    log.info("Pipeline saved → %s/pipeline.joblib", MODELS_DIR)

    # ── Metrics report ─────────────────────────────────────────────────────
    metrics = {
        "baseline_lr_auc": lr_auc,
        "stack_oof_auc": stack_auc,
        "stack_oof_accuracy": stack_acc,
        "stack_oof_f1": stack_f1,
        "optimal_threshold": optimal_thr,
    }
    metrics_df = pd.Series(metrics).to_frame("value")
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"))
    log.info("Metrics saved → %s/metrics.csv", RESULTS_DIR)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn prediction training pipeline")
    parser.add_argument(
        "--tune", action="store_true", default=False,
        help="Run Optuna hyperparameter tuning (slower)",
    )
    parser.add_argument(
        "--trials", type=int, default=OPTUNA_TRIALS,
        help="Number of Optuna trials per model (default: %(default)s)",
    )
    args = parser.parse_args()
    main(args)
