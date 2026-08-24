from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def fbeta_score_manual(
    precision: float,
    recall: float,
    beta: float = 2.0,
) -> float:
    """
    Compute F-beta without relying on predicted labels again.

    beta=2 weights recall more strongly than precision, which is
    appropriate for a pre-MOT warning system where missing a failure
    is more costly than issuing an extra warning.
    """

    if precision == 0 and recall == 0:
        return 0.0

    beta_squared = beta ** 2

    denominator = (
        beta_squared * precision
        + recall
    )

    if denominator == 0:
        return 0.0

    return (
        (1 + beta_squared)
        * precision
        * recall
        / denominator
    )


def metrics_at_threshold(
    frame: pd.DataFrame,
    threshold: float,
) -> dict:
    y_true = frame[
        "actual"
    ].astype(int)

    probability = frame[
        "failure_probability"
    ].astype(float)

    prediction = (
        probability >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        int(value)
        for value in matrix.ravel()
    )

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "threshold":
            float(threshold),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    prediction,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    prediction,
                )
            ),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "specificity":
            float(specificity),

        "f1":
            float(
                f1_score(
                    y_true,
                    prediction,
                    zero_division=0,
                )
            ),

        "f2":
            float(
                fbeta_score_manual(
                    precision,
                    recall,
                    beta=2.0,
                )
            ),

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp,

        "predicted_fail_rate":
            float(
                prediction.mean()
            ),
    }


def build_threshold_table(
    frame: pd.DataFrame,
    minimum: float = 0.05,
    maximum: float = 0.80,
    step: float = 0.01,
) -> pd.DataFrame:
    thresholds = np.arange(
        minimum,
        maximum + (step / 2),
        step,
    )

    rows = [
        metrics_at_threshold(
            frame,
            float(threshold),
        )
        for threshold in thresholds
    ]

    return pd.DataFrame(rows)


def select_max_f2(
    table: pd.DataFrame,
) -> pd.Series:
    """
    Select the threshold that maximises F2.

    If several thresholds tie, prefer the larger threshold
    because it produces fewer unnecessary warnings.
    """

    ordered = table.sort_values(
        by=[
            "f2",
            "threshold",
        ],
        ascending=[
            False,
            False,
        ],
    )

    return ordered.iloc[0]


def select_target_recall(
    table: pd.DataFrame,
    target_recall: float,
) -> pd.Series:
    """
    Among thresholds achieving the desired recall, choose the one
    with highest precision. Ties prefer the higher threshold.

    This creates a useful 'warning threshold' for the application.
    """

    candidates = table[
        table["recall"]
        >= target_recall
    ].copy()

    if candidates.empty:
        # If the requested recall cannot be reached within the
        # threshold search range, select the highest recall found.
        candidates = table[
            table["recall"]
            == table["recall"].max()
        ].copy()

    ordered = candidates.sort_values(
        by=[
            "precision",
            "threshold",
        ],
        ascending=[
            False,
            False,
        ],
    )

    return ordered.iloc[0]


def make_calibration_table(
    frame: pd.DataFrame,
    bins: int = 10,
) -> pd.DataFrame:
    """
    Create an easy-to-read reliability table.

    Each row compares:
        mean predicted failure probability
    with:
        actual fraction of MOT failures.

    Quantile bins are used so each bin contains roughly the same
    number of predictions.
    """

    calibration = frame[
        [
            "actual",
            "failure_probability",
        ]
    ].copy()

    calibration[
        "probability_bin"
    ] = pd.qcut(
        calibration[
            "failure_probability"
        ],
        q=bins,
        duplicates="drop",
    )

    result = (
        calibration
        .groupby(
            "probability_bin",
            observed=True,
        )
        .agg(
            samples=(
                "actual",
                "size",
            ),

            mean_predicted_probability=(
                "failure_probability",
                "mean",
            ),

            actual_failure_rate=(
                "actual",
                "mean",
            ),

            minimum_probability=(
                "failure_probability",
                "min",
            ),

            maximum_probability=(
                "failure_probability",
                "max",
            ),
        )
        .reset_index()
    )

    result[
        "calibration_error"
    ] = (
        result[
            "actual_failure_rate"
        ]
        - result[
            "mean_predicted_probability"
        ]
    )

    # Convert interval objects to strings for clean CSV output.
    result[
        "probability_bin"
    ] = result[
        "probability_bin"
    ].astype(str)

    return result


def print_metrics(
    title: str,
    metrics: dict,
) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)

    print(
        f"Threshold:          "
        f"{metrics['threshold']:.2f}"
    )

    print(
        f"Accuracy:           "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced accuracy:  "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Precision (FAIL):   "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall (FAIL):      "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Specificity (PASS): "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"F1:                 "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"F2:                 "
        f"{metrics['f2']:.4f}"
    )

    print(
        f"Predicted fail rate:"
        f" {metrics['predicted_fail_rate']:.4f}"
    )

    print()
    print(
        "Confusion matrix "
        "(positive class = FAIL)"
    )

    print(
        f"TN={metrics['tn']:,}  "
        f"FP={metrics['fp']:,}  "
        f"FN={metrics['fn']:,}  "
        f"TP={metrics['tp']:,}"
    )


def row_to_dict(
    row: pd.Series,
) -> dict:
    return {
        key: (
            value.item()
            if hasattr(value, "item")
            else value
        )
        for key, value in row.to_dict().items()
    }


def analyse(
    validation_path: Path,
    test_path: Path,
    output_directory: Path,
    target_recall: float,
) -> None:
    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation predictions not found: "
            f"{validation_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test predictions not found: "
            f"{test_path}"
        )

    validation = pd.read_csv(
        validation_path
    )

    test = pd.read_csv(
        test_path
    )

    required = {
        "actual",
        "failure_probability",
    }

    for name, frame in [
        ("validation", validation),
        ("test", test),
    ]:
        missing = (
            required
            - set(frame.columns)
        )

        if missing:
            raise ValueError(
                f"{name} predictions are missing: "
                f"{sorted(missing)}"
            )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_table = (
        build_threshold_table(
            validation
        )
    )

    threshold_table.to_csv(
        output_directory
        / "threshold_analysis_validation.csv",
        index=False,
    )

    max_f2_row = select_max_f2(
        threshold_table
    )

    target_recall_row = (
        select_target_recall(
            threshold_table,
            target_recall=target_recall,
        )
    )

    max_f2_threshold = float(
        max_f2_row[
            "threshold"
        ]
    )

    warning_threshold = float(
        target_recall_row[
            "threshold"
        ]
    )

    print_metrics(
        "VALIDATION: MAXIMUM F2 THRESHOLD",
        row_to_dict(
            max_f2_row
        ),
    )

    print_metrics(
        (
            "VALIDATION: HIGH-RECALL WARNING "
            f"THRESHOLD (target recall >= {target_recall:.0%})"
        ),
        row_to_dict(
            target_recall_row
        ),
    )

    # Apply thresholds chosen ONLY from validation data to the
    # held-out test predictions.
    test_f2_metrics = (
        metrics_at_threshold(
            test,
            max_f2_threshold,
        )
    )

    test_warning_metrics = (
        metrics_at_threshold(
            test,
            warning_threshold,
        )
    )

    print_metrics(
        "TEST: VALIDATION-SELECTED MAX-F2 THRESHOLD",
        test_f2_metrics,
    )

    print_metrics(
        (
            "TEST: VALIDATION-SELECTED "
            "HIGH-RECALL WARNING THRESHOLD"
        ),
        test_warning_metrics,
    )

    calibration = (
        make_calibration_table(
            test,
            bins=10,
        )
    )

    calibration.to_csv(
        output_directory
        / "calibration_table_test.csv",
        index=False,
    )

    print()
    print("=" * 78)
    print("TEST PROBABILITY CALIBRATION TABLE")
    print("=" * 78)

    print(
        calibration[
            [
                "samples",
                "mean_predicted_probability",
                "actual_failure_rate",
                "calibration_error",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    decision_config = {
        "model":
            "random_forest",

        "threshold_selection_dataset":
            "validation",

        "max_f2_threshold":
            max_f2_threshold,

        "high_recall_warning": {
            "target_recall":
                float(
                    target_recall
                ),

            "threshold":
                warning_threshold,

            "validation_metrics":
                row_to_dict(
                    target_recall_row
                ),

            "test_metrics":
                test_warning_metrics,
        },

        "test_metrics_at_max_f2":
            test_f2_metrics,
    }

    with (
        output_directory
        / "decision_threshold.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            decision_config,
            handle,
            indent=2,
        )

    print()
    print(
        "Saved:",
        output_directory
        / "threshold_analysis_validation.csv",
    )

    print(
        "Saved:",
        output_directory
        / "calibration_table_test.csv",
    )

    print(
        "Saved:",
        output_directory
        / "decision_threshold.json",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Tune the MOT FAIL decision threshold on validation "
            "predictions and apply it unchanged to held-out test "
            "predictions."
        )
    )

    parser.add_argument(
        "--validation",
        type=str,
        default=(
            "reports/"
            "validation_predictions_random_forest.csv"
        ),
    )

    parser.add_argument(
        "--test",
        type=str,
        default="reports/test_predictions.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
    )

    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.70,
        help=(
            "Desired FAIL recall for the application warning "
            "threshold. Default: 0.70."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not 0 < args.target_recall <= 1:
        raise ValueError(
            "--target-recall must be between 0 and 1."
        )

    analyse(
        validation_path=Path(
            args.validation
        ),
        test_path=Path(
            args.test
        ),
        output_directory=Path(
            args.output_dir
        ),
        target_recall=(
            args.target_recall
        ),
    )


if __name__ == "__main__":
    main()