from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    RobustScaler,
)


RANDOM_STATE = 42


CATEGORY_TARGETS = {
    "tyres":
        "target_failure_tyres",

    "brakes":
        "target_failure_brakes",

    "suspension":
        "target_failure_suspension",

    "lighting":
        "target_failure_lighting",

    "steering":
        "target_failure_steering",

    "visibility":
        "target_failure_visibility",

    "emissions_exhaust":
        "target_failure_emissions_exhaust",

    "body_structure":
        "target_failure_body_structure",

    "seatbelts_srs":
        "target_failure_seatbelts_srs",

    "other":
        "target_failure_other",
}


CATEGORICAL_FEATURES = [
    "make",
    "model",
    "fuel_type",
]


LOG_NUMERIC_FEATURES = [
    "latest_mileage",
    "average_annual_mileage",
    "mileage_since_previous_mot",
    "days_since_last_mot",
]


EXCLUDED_NON_TARGET_COLUMNS = {
    "registration",
    "history_event_count",
}


def load_dataset(
    csv_path: Path,
) -> pd.DataFrame:
    """
    Load the leakage-free training CSV created by the bulk/delta pipeline.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    df = pd.read_csv(
        csv_path,
        low_memory=False,
    )

    required = {
        "registration",
        "target_failed",
        *CATEGORY_TARGETS.values(),
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    df[
        "target_failed"
    ] = pd.to_numeric(
        df[
            "target_failed"
        ],
        errors="raise",
    ).astype(int)

    for target_column in (
        CATEGORY_TARGETS.values()
    ):
        df[
            target_column
        ] = pd.to_numeric(
            df[
                target_column
            ],
            errors="coerce",
        ).fillna(0).astype(int)

    return df


def get_feature_columns(
    df: pd.DataFrame,
) -> tuple[
    list[str],
    list[str],
    list[str],
]:
    """
    Determine the exact predictors.

    Every target_* column is excluded, preventing accidental label leakage.
    """

    target_columns = {
        column
        for column in df.columns
        if column.startswith(
            "target_"
        )
    }

    excluded = (
        target_columns
        | EXCLUDED_NON_TARGET_COLUMNS
    )

    categorical = [
        column
        for column in CATEGORICAL_FEATURES
        if column in df.columns
        and column not in excluded
    ]

    numeric_candidates = [
        column
        for column in df.columns
        if column not in excluded
        and column not in categorical
    ]

    numeric = []

    for column in numeric_candidates:

        converted = pd.to_numeric(
            df[
                column
            ],
            errors="coerce",
        )

        if converted.notna().any():

            df[
                column
            ] = converted

            numeric.append(
                column
            )

    log_numeric = [
        column
        for column in LOG_NUMERIC_FEATURES
        if column in numeric
    ]

    regular_numeric = [
        column
        for column in numeric
        if column not in log_numeric
    ]

    return (
        categorical,
        log_numeric,
        regular_numeric,
    )


def create_preprocessor(
    categorical_features: list[str],
    log_numeric_features: list[str],
    regular_numeric_features: list[str],
) -> ColumnTransformer:
    """
    Use the same broad preprocessing philosophy as the PASS/FAIL model.
    """

    log_numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "log1p",
                FunctionTransformer(
                    np.log1p,
                    feature_names_out=(
                        "one-to-one"
                    ),
                ),
            ),
            (
                "scaler",
                RobustScaler(),
            ),
        ]
    )

    regular_numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                RobustScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=(
                        "most_frequent"
                    ),
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown=(
                        "infrequent_if_exist"
                    ),
                    min_frequency=20,
                    sparse_output=True,
                ),
            ),
        ]
    )

    transformers = []

    if log_numeric_features:
        transformers.append(
            (
                "log_numeric",
                log_numeric_pipeline,
                log_numeric_features,
            )
        )

    if regular_numeric_features:
        transformers.append(
            (
                "numeric",
                regular_numeric_pipeline,
                regular_numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )


def make_vehicle_splits(
    failed_df: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Split failed-MOT examples by registration.

    Approximate proportions:
        60% train
        20% validation
        20% test

    A registration can appear in only one split.
    """

    groups = failed_df[
        "registration"
    ].astype(str)

    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_validation_index, test_index = next(
        outer.split(
            failed_df,
            groups=groups,
        )
    )

    train_validation_df = (
        failed_df.iloc[
            train_validation_index
        ]
    )

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=(
            RANDOM_STATE + 1
        ),
    )

    inner_train_relative, validation_relative = next(
        inner.split(
            train_validation_df,
            groups=(
                train_validation_df[
                    "registration"
                ].astype(str)
            ),
        )
    )

    train_index = (
        train_validation_index[
            inner_train_relative
        ]
    )

    validation_index = (
        train_validation_index[
            validation_relative
        ]
    )

    train_groups = set(
        failed_df.iloc[
            train_index
        ][
            "registration"
        ].astype(str)
    )

    validation_groups = set(
        failed_df.iloc[
            validation_index
        ][
            "registration"
        ].astype(str)
    )

    test_groups = set(
        failed_df.iloc[
            test_index
        ][
            "registration"
        ].astype(str)
    )

    assert train_groups.isdisjoint(
        validation_groups
    )

    assert train_groups.isdisjoint(
        test_groups
    )

    assert validation_groups.isdisjoint(
        test_groups
    )

    return (
        train_index,
        validation_index,
        test_index,
    )


def print_split_summary(
    name: str,
    frame: pd.DataFrame,
) -> None:
    print(
        f"{name:<12} "
        f"rows={len(frame):>6,}  "
        f"vehicles="
        f"{frame['registration'].nunique():>5,}"
    )


def print_category_prevalence(
    frame: pd.DataFrame,
    title: str,
) -> None:
    print()
    print(title)
    print(
        "-" * len(title)
    )

    for category, column in (
        CATEGORY_TARGETS.items()
    ):

        rate = (
            frame[
                column
            ].mean()
            * 100
        )

        count = int(
            frame[
                column
            ].sum()
        )

        print(
            f"{category:<20} "
            f"{count:>5,} "
            f"({rate:>6.2f}%)"
        )


def build_model(
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """
    Multi-label logistic regression.

    Each category receives its own binary LogisticRegression classifier.

    class_weight='balanced' helps less common categories such as
    seatbelt/SRS without creating synthetic examples.
    """

    category_classifier = (
        OneVsRestClassifier(
            LogisticRegression(
                max_iter=2500,
                solver="liblinear",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            n_jobs=-1,
        )
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                category_classifier,
            ),
        ]
    )


def probability_matrix(
    model: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Return shape:
        rows x categories
    """

    probabilities = (
        model.predict_proba(
            X
        )
    )

    probabilities = np.asarray(
        probabilities
    )

    if probabilities.ndim != 2:
        raise RuntimeError(
            "Unexpected category probability shape: "
            f"{probabilities.shape}"
        )

    return probabilities


def evaluate_categories(
    y_true: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict:
    """
    Evaluate every failure category separately.

    The 0.50 threshold is only a baseline operating point.
    Later we can tune a separate threshold per category.
    """

    results = {}

    pr_auc_values = []
    roc_auc_values = []
    f1_values = []

    category_names = list(
        CATEGORY_TARGETS.keys()
    )

    target_columns = list(
        CATEGORY_TARGETS.values()
    )

    for index, (
        category,
        target_column,
    ) in enumerate(
        zip(
            category_names,
            target_columns,
        )
    ):

        actual = (
            y_true[
                target_column
            ].astype(int)
        )

        probability = (
            probabilities[
                :,
                index
            ]
        )

        predicted = (
            probability
            >= threshold
        ).astype(int)

        prevalence = float(
            actual.mean()
        )

        positive_count = int(
            actual.sum()
        )

        precision = float(
            precision_score(
                actual,
                predicted,
                zero_division=0,
            )
        )

        recall = float(
            recall_score(
                actual,
                predicted,
                zero_division=0,
            )
        )

        f1 = float(
            f1_score(
                actual,
                predicted,
                zero_division=0,
            )
        )

        if (
            actual.nunique()
            >= 2
        ):
            roc_auc = float(
                roc_auc_score(
                    actual,
                    probability,
                )
            )

            pr_auc = float(
                average_precision_score(
                    actual,
                    probability,
                )
            )

            brier = float(
                brier_score_loss(
                    actual,
                    probability,
                )
            )

            roc_auc_values.append(
                roc_auc
            )

            pr_auc_values.append(
                pr_auc
            )

        else:
            roc_auc = None
            pr_auc = None
            brier = None

        f1_values.append(
            f1
        )

        results[
            category
        ] = {
            "positive_count":
                positive_count,

            "prevalence":
                prevalence,

            "precision_at_0_5":
                precision,

            "recall_at_0_5":
                recall,

            "f1_at_0_5":
                f1,

            "roc_auc":
                roc_auc,

            "pr_auc":
                pr_auc,

            "brier_score":
                brier,
        }

    results[
        "_macro"
    ] = {
        "mean_roc_auc": (
            float(
                np.mean(
                    roc_auc_values
                )
            )
            if roc_auc_values
            else None
        ),

        "mean_pr_auc": (
            float(
                np.mean(
                    pr_auc_values
                )
            )
            if pr_auc_values
            else None
        ),

        "mean_f1_at_0_5":
            float(
                np.mean(
                    f1_values
                )
            ),
    }

    return results


def print_evaluation(
    results: dict,
) -> None:
    print()

    print(
        f"{'Category':<20}"
        f"{'Prev':>8}"
        f"{'ROC-AUC':>10}"
        f"{'PR-AUC':>10}"
        f"{'Prec':>9}"
        f"{'Recall':>9}"
        f"{'F1':>9}"
    )

    print(
        "-" * 75
    )

    for category in (
        CATEGORY_TARGETS.keys()
    ):

        metrics = (
            results[
                category
            ]
        )

        roc_auc = (
            "-"
            if metrics[
                "roc_auc"
            ] is None
            else (
                f"{metrics['roc_auc']:.3f}"
            )
        )

        pr_auc = (
            "-"
            if metrics[
                "pr_auc"
            ] is None
            else (
                f"{metrics['pr_auc']:.3f}"
            )
        )

        print(
            f"{category:<20}"
            f"{metrics['prevalence']:>8.3f}"
            f"{roc_auc:>10}"
            f"{pr_auc:>10}"
            f"{metrics['precision_at_0_5']:>9.3f}"
            f"{metrics['recall_at_0_5']:>9.3f}"
            f"{metrics['f1_at_0_5']:>9.3f}"
        )

    macro = results[
        "_macro"
    ]

    print()
    print(
        "Macro mean ROC-AUC:",
        (
            f"{macro['mean_roc_auc']:.4f}"
            if macro[
                "mean_roc_auc"
            ] is not None
            else "-"
        ),
    )

    print(
        "Macro mean PR-AUC:",
        (
            f"{macro['mean_pr_auc']:.4f}"
            if macro[
                "mean_pr_auc"
            ] is not None
            else "-"
        ),
    )

    print(
        "Macro mean F1 @ 0.50:",
        f"{macro['mean_f1_at_0_5']:.4f}",
    )


def save_prediction_file(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    destination: Path,
) -> None:
    output = frame[
        [
            "registration",
            "target_date",
        ]
    ].copy()

    for index, category in enumerate(
        CATEGORY_TARGETS.keys()
    ):

        target_column = (
            CATEGORY_TARGETS[
                category
            ]
        )

        output[
            f"actual_{category}"
        ] = (
            frame[
                target_column
            ].astype(int)
        )

        output[
            f"probability_{category}"
        ] = (
            probabilities[
                :,
                index
            ]
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        destination,
        index=False,
    )


def save_json(
    data: dict,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            data,
            handle,
            indent=2,
        )


def train_category_model(
    input_csv: Path,
    model_directory: Path,
    report_directory: Path,
) -> None:
    full_df = load_dataset(
        input_csv
    )

    # Conditional model:
    # learn P(category | initial MOT failed)
    failed_df = (
        full_df[
            full_df[
                "target_failed"
            ] == 1
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        "Complete dataset rows:",
        f"{len(full_df):,}",
    )

    print(
        "Failed-MOT rows used for category modelling:",
        f"{len(failed_df):,}",
    )

    print(
        "Unique vehicles with failed examples:",
        f"{failed_df['registration'].nunique():,}",
    )

    print_category_prevalence(
        failed_df,
        "CATEGORY PREVALENCE AMONG FAILED MOTs",
    )

    (
        categorical_features,
        log_numeric_features,
        regular_numeric_features,
    ) = get_feature_columns(
        failed_df
    )

    feature_columns = (
        log_numeric_features
        + regular_numeric_features
        + categorical_features
    )

    (
        train_index,
        validation_index,
        test_index,
    ) = make_vehicle_splits(
        failed_df
    )

    train_df = failed_df.iloc[
        train_index
    ].copy()

    validation_df = failed_df.iloc[
        validation_index
    ].copy()

    test_df = failed_df.iloc[
        test_index
    ].copy()

    print()
    print(
        "Vehicle-level split of failed MOT examples:"
    )

    print_split_summary(
        "TRAIN",
        train_df,
    )

    print_split_summary(
        "VALIDATION",
        validation_df,
    )

    print_split_summary(
        "TEST",
        test_df,
    )

    print_category_prevalence(
        train_df,
        "TRAIN CATEGORY PREVALENCE",
    )

    print_category_prevalence(
        validation_df,
        "VALIDATION CATEGORY PREVALENCE",
    )

    print_category_prevalence(
        test_df,
        "TEST CATEGORY PREVALENCE",
    )

    target_columns = list(
        CATEGORY_TARGETS.values()
    )

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        target_columns
    ]

    X_validation = validation_df[
        feature_columns
    ]

    y_validation = validation_df[
        target_columns
    ]

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        target_columns
    ]

    preprocessor = create_preprocessor(
        categorical_features=(
            categorical_features
        ),
        log_numeric_features=(
            log_numeric_features
        ),
        regular_numeric_features=(
            regular_numeric_features
        ),
    )

    model = build_model(
        preprocessor
    )

    print()
    print(
        "=" * 75
    )

    print(
        "TRAINING MULTI-LABEL CATEGORY MODEL"
    )

    print(
        "=" * 75
    )

    model.fit(
        X_train,
        y_train,
    )

    validation_probabilities = (
        probability_matrix(
            model,
            X_validation,
        )
    )

    validation_results = (
        evaluate_categories(
            y_validation,
            validation_probabilities,
            threshold=0.50,
        )
    )

    print()
    print(
        "=" * 75
    )

    print(
        "VALIDATION RESULTS"
    )

    print(
        "=" * 75
    )

    print_evaluation(
        validation_results
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_prediction_file(
        validation_df,
        validation_probabilities,
        report_directory
        / "validation_category_predictions.csv",
    )

    # Refit on train + validation after the modelling approach
    # has been evaluated on validation.
    train_validation_df = (
        pd.concat(
            [
                train_df,
                validation_df,
            ],
            axis=0,
        )
    )

    X_train_validation = (
        train_validation_df[
            feature_columns
        ]
    )

    y_train_validation = (
        train_validation_df[
            target_columns
        ]
    )

    final_model = build_model(
        create_preprocessor(
            categorical_features=(
                categorical_features
            ),
            log_numeric_features=(
                log_numeric_features
            ),
            regular_numeric_features=(
                regular_numeric_features
            ),
        )
    )

    print()
    print(
        "Refitting category model on TRAIN + VALIDATION..."
    )

    final_model.fit(
        X_train_validation,
        y_train_validation,
    )

    test_probabilities = (
        probability_matrix(
            final_model,
            X_test,
        )
    )

    test_results = (
        evaluate_categories(
            y_test,
            test_probabilities,
            threshold=0.50,
        )
    )

    print()
    print(
        "=" * 75
    )

    print(
        "FINAL HELD-OUT TEST RESULTS"
    )

    print(
        "=" * 75
    )

    print_evaluation(
        test_results
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        model_directory
        / "mot_failure_category_model.joblib"
    )

    joblib.dump(
        final_model,
        model_path,
    )

    save_prediction_file(
        test_df,
        test_probabilities,
        report_directory
        / "test_category_predictions.csv",
    )

    configuration = {
        "model_type":
            "conditional_multilabel_logistic_regression",

        "interpretation":
            (
                "Each output estimates the probability of a "
                "failure category conditional on the initial MOT "
                "test failing."
            ),

        "categories":
            list(
                CATEGORY_TARGETS.keys()
            ),

        "target_columns":
            CATEGORY_TARGETS,

        "feature_columns":
            feature_columns,

        "categorical_features":
            categorical_features,

        "log_numeric_features":
            log_numeric_features,

        "regular_numeric_features":
            regular_numeric_features,
    }

    save_json(
        configuration,
        model_directory
        / "category_model_features.json",
    )

    complete_report = {
        "dataset": {
            "all_rows":
                int(
                    len(
                        full_df
                    )
                ),

            "failed_rows":
                int(
                    len(
                        failed_df
                    )
                ),

            "unique_failed_vehicles":
                int(
                    failed_df[
                        "registration"
                    ].nunique()
                ),
        },

        "split": {
            "train_rows":
                int(
                    len(
                        train_df
                    )
                ),

            "validation_rows":
                int(
                    len(
                        validation_df
                    )
                ),

            "test_rows":
                int(
                    len(
                        test_df
                    )
                ),

            "train_vehicles":
                int(
                    train_df[
                        "registration"
                    ].nunique()
                ),

            "validation_vehicles":
                int(
                    validation_df[
                        "registration"
                    ].nunique()
                ),

            "test_vehicles":
                int(
                    test_df[
                        "registration"
                    ].nunique()
                ),
        },

        "validation_metrics":
            validation_results,

        "test_metrics":
            test_results,

        "model_path":
            str(
                model_path
            ),
    }

    save_json(
        complete_report,
        report_directory
        / "category_model_metrics.json",
    )

    print()
    print(
        "=" * 75
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 75
    )

    print(
        "Model:",
        model_path,
    )

    print(
        "Features:",
        model_directory
        / "category_model_features.json",
    )

    print(
        "Metrics:",
        report_directory
        / "category_model_metrics.json",
    )

    print(
        "Test predictions:",
        report_directory
        / "test_category_predictions.csv",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a multi-label model to predict MOT "
            "failure categories conditional on an initial MOT failure."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=(
            "data/processed/"
            "mot_training_sample.csv"
        ),
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
    )

    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_category_model(
        input_csv=Path(
            args.input
        ),
        model_directory=Path(
            args.model_dir
        ),
        report_directory=Path(
            args.report_dir
        ),
    )


if __name__ == "__main__":
    main()