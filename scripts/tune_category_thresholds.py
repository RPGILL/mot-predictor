from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


CATEGORIES = [
    "tyres",
    "brakes",
    "suspension",
    "lighting",
    "steering",
    "visibility",
    "emissions_exhaust",
    "body_structure",
    "seatbelts_srs",
    "other",
]


def metrics_at_threshold(
    actual: pd.Series,
    probability: pd.Series,
    threshold: float,
) -> dict:
    y_true = actual.astype(int)

    predicted = (
        probability.astype(float)
        >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        int(value)
        for value in matrix.ravel()
    )

    precision = float(
        precision_score(
            y_true,
            predicted,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_true,
            predicted,
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            y_true,
            predicted,
            zero_division=0,
        )
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "threshold":
            float(threshold),

        "precision":
            precision,

        "recall":
            recall,

        "specificity":
            float(specificity),

        "f1":
            f1,

        "predicted_positive_rate":
            float(
                predicted.mean()
            ),

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp,
    }


def build_threshold_table(
    actual: pd.Series,
    probability: pd.Series,
    minimum: float = 0.05,
    maximum: float = 0.95,
    step: float = 0.01,
) -> pd.DataFrame:
    thresholds = np.arange(
        minimum,
        maximum + step / 2,
        step,
    )

    rows = [
        metrics_at_threshold(
            actual,
            probability,
            float(threshold),
        )
        for threshold in thresholds
    ]

    return pd.DataFrame(rows)


def select_target_recall_threshold(
    table: pd.DataFrame,
    target_recall: float,
) -> pd.Series:
    """
    Choose the highest-precision threshold that reaches the
    requested recall.

    Ties prefer the higher threshold, producing fewer warnings.
    """

    candidates = table[
        table["recall"]
        >= target_recall
    ].copy()

    if candidates.empty:
        candidates = table[
            table["recall"]
            == table["recall"].max()
        ].copy()

    candidates = candidates.sort_values(
        by=[
            "precision",
            "threshold",
        ],
        ascending=[
            False,
            False,
        ],
    )

    return candidates.iloc[0]


def select_max_f1_threshold(
    table: pd.DataFrame,
) -> pd.Series:
    candidates = table.sort_values(
        by=[
            "f1",
            "threshold",
        ],
        ascending=[
            False,
            False,
        ],
    )

    return candidates.iloc[0]


def row_to_dict(
    row: pd.Series,
) -> dict:
    result = {}

    for key, value in row.to_dict().items():
        if hasattr(
            value,
            "item",
        ):
            value = value.item()

        result[key] = value

    return result


def validate_prediction_file(
    frame: pd.DataFrame,
    filename: str,
) -> None:
    required = set()

    for category in CATEGORIES:
        required.add(
            f"actual_{category}"
        )

        required.add(
            f"probability_{category}"
        )

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{filename} is missing columns: "
            f"{sorted(missing)}"
        )


def print_category_result(
    category: str,
    validation_metrics: dict,
    test_metrics: dict,
) -> None:
    print()
    print(
        f"{category.upper()}"
    )

    print(
        "-" * 76
    )

    print(
        "Validation-selected threshold:",
        f"{validation_metrics['threshold']:.2f}",
    )

    print(
        "VALIDATION  "
        f"precision={validation_metrics['precision']:.3f}  "
        f"recall={validation_metrics['recall']:.3f}  "
        f"F1={validation_metrics['f1']:.3f}  "
        f"flag_rate={validation_metrics['predicted_positive_rate']:.3f}"
    )

    print(
        "TEST        "
        f"precision={test_metrics['precision']:.3f}  "
        f"recall={test_metrics['recall']:.3f}  "
        f"F1={test_metrics['f1']:.3f}  "
        f"flag_rate={test_metrics['predicted_positive_rate']:.3f}"
    )

    print(
        "TEST confusion: "
        f"TN={test_metrics['tn']:,}  "
        f"FP={test_metrics['fp']:,}  "
        f"FN={test_metrics['fn']:,}  "
        f"TP={test_metrics['tp']:,}"
    )


def tune(
    validation_path: Path,
    test_path: Path,
    output_directory: Path,
    target_recall: float,
) -> None:
    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation file not found: "
            f"{validation_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test file not found: "
            f"{test_path}"
        )

    validation = pd.read_csv(
        validation_path
    )

    test = pd.read_csv(
        test_path
    )

    validate_prediction_file(
        validation,
        str(
            validation_path
        ),
    )

    validate_prediction_file(
        test,
        str(
            test_path
        ),
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    configuration = {
        "selection_dataset":
            "validation",

        "selection_rule":
            (
                "highest precision among thresholds "
                "meeting the requested recall"
            ),

        "target_recall":
            float(
                target_recall
            ),

        "important_note":
            (
                "The category classifier was trained with "
                "class_weight='balanced'. Its raw outputs are used "
                "as ranking/decision scores here, not claimed as "
                "calibrated probabilities."
            ),

        "categories": {},
    }

    all_threshold_rows = []

    print()
    print(
        "=" * 76
    )

    print(
        "CATEGORY-SPECIFIC WARNING THRESHOLDS"
    )

    print(
        "=" * 76
    )

    for category in CATEGORIES:

        actual_column = (
            f"actual_{category}"
        )

        probability_column = (
            f"probability_{category}"
        )

        validation_table = (
            build_threshold_table(
                validation[
                    actual_column
                ],
                validation[
                    probability_column
                ],
            )
        )

        warning_row = (
            select_target_recall_threshold(
                validation_table,
                target_recall=(
                    target_recall
                ),
            )
        )

        best_f1_row = (
            select_max_f1_threshold(
                validation_table
            )
        )

        warning_threshold = float(
            warning_row[
                "threshold"
            ]
        )

        validation_warning_metrics = (
            row_to_dict(
                warning_row
            )
        )

        test_warning_metrics = (
            metrics_at_threshold(
                test[
                    actual_column
                ],
                test[
                    probability_column
                ],
                warning_threshold,
            )
        )

        configuration[
            "categories"
        ][
            category
        ] = {
            "warning_threshold":
                warning_threshold,

            "validation_metrics":
                validation_warning_metrics,

            "test_metrics":
                test_warning_metrics,

            "max_f1_threshold_validation":
                float(
                    best_f1_row[
                        "threshold"
                    ]
                ),

            "max_f1_validation_metrics":
                row_to_dict(
                    best_f1_row
                ),
        }

        print_category_result(
            category,
            validation_warning_metrics,
            test_warning_metrics,
        )

        category_table = (
            validation_table.copy()
        )

        category_table.insert(
            0,
            "category",
            category,
        )

        all_threshold_rows.append(
            category_table
        )

    threshold_frame = pd.concat(
        all_threshold_rows,
        ignore_index=True,
    )

    threshold_frame.to_csv(
        output_directory
        / "category_threshold_analysis_validation.csv",
        index=False,
    )

    config_path = (
        output_directory
        / "category_thresholds.json"
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            configuration,
            handle,
            indent=2,
        )

    print()
    print(
        "=" * 76
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 76
    )

    print(
        output_directory
        / "category_threshold_analysis_validation.csv"
    )

    print(
        config_path
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Tune one validation-selected warning threshold "
            "for each MOT failure category."
        )
    )

    parser.add_argument(
        "--validation",
        type=str,
        default=(
            "reports/"
            "validation_category_predictions.csv"
        ),
    )

    parser.add_argument(
        "--test",
        type=str,
        default=(
            "reports/"
            "test_category_predictions.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
    )

    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.60,
        help=(
            "Desired recall per category for the warning "
            "threshold. Default: 0.60."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not 0 < args.target_recall <= 1:
        raise ValueError(
            "--target-recall must be between 0 and 1."
        )

    tune(
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