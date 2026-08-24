from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler


RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["make", "model", "fuel_type"]

# Highly skewed non-negative features. log1p reduces the effect of extreme values.
LOG_NUMERIC_FEATURES = [
    "latest_mileage",
    "average_annual_mileage",
    "mileage_since_previous_mot",
    "days_since_last_mot",
]

# Metadata / duplicate bookkeeping columns that should not enter the model.
EXCLUDED_NON_TARGET_COLUMNS = {
    "registration",
    "history_event_count",  # essentially duplicates mot_event_count
}


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    required = {"registration", "target_failed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["target_failed"] = pd.to_numeric(
        df["target_failed"], errors="raise"
    ).astype(int)

    return df


def get_feature_columns(df: pd.DataFrame):
    """Return categorical, log-numeric and regular numeric feature lists."""

    # Every target_* column is excluded automatically to prevent label leakage.
    target_columns = {c for c in df.columns if c.startswith("target_")}
    excluded = target_columns | EXCLUDED_NON_TARGET_COLUMNS

    categorical = [
        c for c in CATEGORICAL_FEATURES
        if c in df.columns and c not in excluded
    ]

    numeric = []
    for column in df.columns:
        if column in excluded or column in categorical:
            continue

        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().any():
            df[column] = converted
            numeric.append(column)

    log_numeric = [c for c in LOG_NUMERIC_FEATURES if c in numeric]
    regular_numeric = [c for c in numeric if c not in log_numeric]

    return categorical, log_numeric, regular_numeric


def create_preprocessor(categorical, log_numeric, regular_numeric):
    """
    Preprocessing is fitted only on training data.

    - missing numeric values -> median
    - mileage/time variables -> log1p + robust scaling
    - remaining numeric features -> robust scaling
    - categoricals -> most-frequent imputation + one-hot encoding
    - rare categories -> grouped automatically
    """

    log_numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "log1p",
            FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        ),
        ("scaler", RobustScaler()),
    ])

    regular_numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=20,
                sparse_output=True,
            ),
        ),
    ])

    transformers = []
    if log_numeric:
        transformers.append(("log_numeric", log_numeric_pipeline, log_numeric))
    if regular_numeric:
        transformers.append(("numeric", regular_numeric_pipeline, regular_numeric))
    if categorical:
        transformers.append(("categorical", categorical_pipeline, categorical))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )


def make_group_splits(df: pd.DataFrame):
    """
    Create approximately 60/20/20 train/validation/test splits.

    Registrations never overlap between splits. StratifiedGroupKFold also tries
    to preserve the PASS/FAIL distribution while respecting those groups.
    """

    y = df["target_failed"].to_numpy()
    groups = df["registration"].astype(str).to_numpy()

    # One of five folds becomes the final test set (~20%).
    outer = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_val_idx, test_idx = next(
        outer.split(np.zeros(len(df)), y, groups)
    )

    train_val_df = df.iloc[train_val_idx]
    y_tv = train_val_df["target_failed"].to_numpy()
    groups_tv = train_val_df["registration"].astype(str).to_numpy()

    # One of four folds from the remaining data becomes validation (~20% total).
    inner = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE + 1,
    )

    train_rel, val_rel = next(
        inner.split(np.zeros(len(train_val_df)), y_tv, groups_tv)
    )

    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]

    train_regs = set(df.iloc[train_idx]["registration"].astype(str))
    val_regs = set(df.iloc[val_idx]["registration"].astype(str))
    test_regs = set(df.iloc[test_idx]["registration"].astype(str))

    assert train_regs.isdisjoint(val_regs)
    assert train_regs.isdisjoint(test_regs)
    assert val_regs.isdisjoint(test_regs)

    return train_idx, val_idx, test_idx


def print_split_summary(name: str, subset: pd.DataFrame):
    failure_rate = subset["target_failed"].mean() * 100
    print(
        f"{name:<12} rows={len(subset):>7,}  "
        f"vehicles={subset['registration'].nunique():>5,}  "
        f"fail_rate={failure_rate:>6.2f}%"
    )


def build_models(preprocessor):
    return {
        "dummy_prior": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", DummyClassifier(strategy="prior")),
        ]),
        "logistic_regression": Pipeline([
            ("preprocessor", clone(preprocessor)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", clone(preprocessor)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=350,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
    }


def evaluate_model(model, X: pd.DataFrame, y: pd.Series):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    cm = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc_average_precision": float(
            average_precision_score(y, probabilities)
        ),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        "classification_report": classification_report(
            y,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    prediction_frame = pd.DataFrame(
        {
            "actual": y.to_numpy(),
            "predicted": predictions,
            "failure_probability": probabilities,
        },
        index=y.index,
    )

    return metrics, prediction_frame


def print_metrics(name: str, metrics: dict):
    print("\n" + "-" * 72)
    print(name)
    print("-" * 72)
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision (FAIL):  {metrics['precision']:.4f}")
    print(f"Recall (FAIL):     {metrics['recall']:.4f}")
    print(f"F1 (FAIL):         {metrics['f1']:.4f}")
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:            {metrics['pr_auc_average_precision']:.4f}")
    print(f"Brier score:       {metrics['brier_score']:.4f}")

    cm = metrics["confusion_matrix"]
    print("\nConfusion matrix (positive class = FAIL)")
    print(
        f"TN={cm['true_negative']:,}  "
        f"FP={cm['false_positive']:,}  "
        f"FN={cm['false_negative']:,}  "
        f"TP={cm['true_positive']:,}"
    )


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def train(csv_path: Path, model_dir: Path, report_dir: Path):
    df = load_dataset(csv_path)

    categorical, log_numeric, regular_numeric = get_feature_columns(df)
    feature_columns = log_numeric + regular_numeric + categorical

    print(f"Dataset rows: {len(df):,}")
    print(f"Unique vehicles: {df['registration'].nunique():,}")
    print(f"Model features: {len(feature_columns)}")
    print("Categorical features:", categorical)
    print("Log-transformed numeric features:", log_numeric)

    train_idx, val_idx, test_idx = make_group_splits(df)

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    print("\nVehicle-level split:")
    print_split_summary("TRAIN", train_df)
    print_split_summary("VALIDATION", val_df)
    print_split_summary("TEST", test_df)

    X_train = train_df[feature_columns]
    y_train = train_df["target_failed"]
    X_val = val_df[feature_columns]
    y_val = val_df["target_failed"]
    X_test = test_df[feature_columns]
    y_test = test_df["target_failed"]

    preprocessor = create_preprocessor(
        categorical,
        log_numeric,
        regular_numeric,
    )
    models = build_models(preprocessor)

    report_dir.mkdir(parents=True, exist_ok=True)
    validation_results = {}

    print("\n" + "=" * 72)
    print("VALIDATION RESULTS")
    print("=" * 72)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        metrics, predictions = evaluate_model(model, X_val, y_val)
        validation_results[name] = metrics
        print_metrics(name, metrics)

        pred_out = val_df[["registration", "target_date"]].copy()
        pred_out["actual"] = predictions["actual"]
        pred_out["predicted"] = predictions["predicted"]
        pred_out["failure_probability"] = predictions["failure_probability"]
        pred_out.to_csv(
            report_dir / f"validation_predictions_{name}.csv",
            index=False,
        )

    best_name = max(
        validation_results,
        key=lambda name: validation_results[name]["roc_auc"],
    )

    print("\n" + "=" * 72)
    print("SELECTED MODEL")
    print("=" * 72)
    print(
        f"Best validation ROC-AUC: {best_name} "
        f"({validation_results[best_name]['roc_auc']:.4f})"
    )

    # Final fit uses train + validation; test remains untouched until now.
    train_val_df = pd.concat([train_df, val_df], axis=0)
    X_train_val = train_val_df[feature_columns]
    y_train_val = train_val_df["target_failed"]

    best_model = clone(models[best_name])
    print("\nRefitting selected model on TRAIN + VALIDATION...")
    best_model.fit(X_train_val, y_train_val)

    test_metrics, test_predictions = evaluate_model(
        best_model,
        X_test,
        y_test,
    )

    print("\n" + "=" * 72)
    print("FINAL HELD-OUT TEST RESULTS")
    print("=" * 72)
    print_metrics(best_name, test_metrics)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "mot_pass_fail_model.joblib"
    joblib.dump(best_model, model_path)

    test_out = test_df[["registration", "target_date"]].copy()
    test_out["actual"] = test_predictions["actual"]
    test_out["predicted"] = test_predictions["predicted"]
    test_out["failure_probability"] = test_predictions["failure_probability"]
    test_out.to_csv(report_dir / "test_predictions.csv", index=False)

    complete_report = {
        "selected_model": best_name,
        "selection_metric": "validation_roc_auc",
        "dataset": {
            "rows": int(len(df)),
            "unique_vehicles": int(df["registration"].nunique()),
            "overall_failure_rate": float(df["target_failed"].mean()),
        },
        "split": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_vehicles": int(train_df["registration"].nunique()),
            "validation_vehicles": int(val_df["registration"].nunique()),
            "test_vehicles": int(test_df["registration"].nunique()),
        },
        "features": {
            "categorical": categorical,
            "log_numeric": log_numeric,
            "regular_numeric": regular_numeric,
        },
        "validation_metrics": validation_results,
        "test_metrics": test_metrics,
        "model_path": str(model_path),
    }

    save_json(complete_report, report_dir / "model_metrics.json")
    save_json(
        {
            "feature_columns": feature_columns,
            "categorical_features": categorical,
            "log_numeric_features": log_numeric,
            "regular_numeric_features": regular_numeric,
        },
        model_dir / "model_features.json",
    )

    print("\n" + "=" * 72)
    print("FILES SAVED")
    print("=" * 72)
    print("Model:", model_path)
    print("Metrics:", report_dir / "model_metrics.json")
    print("Test predictions:", report_dir / "test_predictions.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train baseline MOT PASS/FAIL models using a "
            "registration-level split."
        )
    )
    parser.add_argument(
        "--input",
        default="data/processed/mot_training_sample.csv",
    )
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--report-dir", default="reports")
    return parser.parse_args()


def main():
    args = parse_args()
    train(
        csv_path=Path(args.input),
        model_dir=Path(args.model_dir),
        report_dir=Path(args.report_dir),
    )


if __name__ == "__main__":
    main()