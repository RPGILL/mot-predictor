from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGET_CATEGORY_COLUMNS = [
    "target_failure_tyres",
    "target_failure_brakes",
    "target_failure_suspension",
    "target_failure_lighting",
    "target_failure_steering",
    "target_failure_visibility",
    "target_failure_emissions_exhaust",
    "target_failure_body_structure",
    "target_failure_seatbelts_srs",
    "target_failure_other",
]


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def safe_percent(part: int, whole: int) -> float:
    if whole == 0:
        return 0.0
    return (part / whole) * 100


def inspect_dataset(csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    print(f"Loading: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    print_section("1. BASIC DATASET INFORMATION")
    print("Rows:", f"{len(df):,}")
    print("Columns:", len(df.columns))

    if "registration" in df.columns:
        print(
            "Unique registrations:",
            f"{df['registration'].nunique(dropna=True):,}",
        )

    if "target_date" in df.columns:
        target_dates = pd.to_datetime(
            df["target_date"],
            errors="coerce",
            utc=True,
        )

        if target_dates.notna().any():
            print("Earliest target date:", target_dates.min())
            print("Latest target date:", target_dates.max())

    print_section("2. TARGET PASS / FAIL BALANCE")

    if "target_failed" not in df.columns:
        print("ERROR: target_failed column is missing.")
    else:
        target_counts = (
            df["target_failed"]
            .value_counts(dropna=False)
            .sort_index()
        )

        total = len(df)
        passes = int(target_counts.get(0, 0))
        failures = int(target_counts.get(1, 0))

        print(
            "PASS targets (0):",
            f"{passes:,}",
            f"({safe_percent(passes, total):.2f}%)",
        )
        print(
            "FAIL targets (1):",
            f"{failures:,}",
            f"({safe_percent(failures, total):.2f}%)",
        )

        other_targets = total - passes - failures

        if other_targets:
            print(
                "Unexpected/missing target values:",
                f"{other_targets:,}",
            )

    print_section("3. FAILURE CATEGORY TARGETS")

    available_target_columns = [
        column
        for column in TARGET_CATEGORY_COLUMNS
        if column in df.columns
    ]

    if not available_target_columns:
        print("No target_failure_* columns found.")
    else:
        for column in available_target_columns:
            count = int(
                pd.to_numeric(
                    df[column],
                    errors="coerce",
                )
                .fillna(0)
                .eq(1)
                .sum()
            )

            print(f"{column}: {count:,}")

    print_section("4. MISSING VALUES")

    missing = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
    else:
        print("Columns with missing values:")

        for column, count in missing.items():
            print(
                f"{column}: {count:,} "
                f"({safe_percent(int(count), len(df)):.2f}%)"
            )

    print_section("5. DUPLICATE / LEAKAGE CHECKS")

    duplicate_rows = int(df.duplicated().sum())

    print(
        "Exact duplicate rows:",
        f"{duplicate_rows:,}",
    )

    if {
        "registration",
        "target_date",
    }.issubset(df.columns):
        duplicate_targets = int(
            df.duplicated(
                subset=[
                    "registration",
                    "target_date",
                ],
                keep=False,
            ).sum()
        )

        print(
            "Rows sharing the same "
            "(registration, target_date):",
            f"{duplicate_targets:,}",
        )

    if {
        "registration",
        "history_event_count",
    }.issubset(df.columns):
        duplicate_history_steps = int(
            df.duplicated(
                subset=[
                    "registration",
                    "history_event_count",
                ],
                keep=False,
            ).sum()
        )

        print(
            "Rows sharing the same "
            "(registration, history_event_count):",
            f"{duplicate_history_steps:,}",
        )

    print_section("6. SANITY CHECKS")

    checks = []

    if "vehicle_age_years" in df.columns:
        age = pd.to_numeric(
            df["vehicle_age_years"],
            errors="coerce",
        )

        checks.append(
            (
                "Negative vehicle ages",
                int((age < 0).sum()),
            )
        )
        checks.append(
            (
                "Vehicle age > 100 years",
                int((age > 100).sum()),
            )
        )

    if "latest_mileage" in df.columns:
        mileage = pd.to_numeric(
            df["latest_mileage"],
            errors="coerce",
        )

        checks.append(
            (
                "Negative latest mileage",
                int((mileage < 0).sum()),
            )
        )
        checks.append(
            (
                "Latest mileage > 1,000,000",
                int((mileage > 1_000_000).sum()),
            )
        )

    if "mileage_since_previous_mot" in df.columns:
        mileage_change = pd.to_numeric(
            df["mileage_since_previous_mot"],
            errors="coerce",
        )

        checks.append(
            (
                "Negative mileage since previous MOT",
                int((mileage_change < 0).sum()),
            )
        )

    if "days_since_last_mot" in df.columns:
        days = pd.to_numeric(
            df["days_since_last_mot"],
            errors="coerce",
        )

        checks.append(
            (
                "Negative days since last MOT",
                int((days < 0).sum()),
            )
        )
        checks.append(
            (
                "Days since last MOT > 1,500",
                int((days > 1500).sum()),
            )
        )

    if not checks:
        print("No sanity-check columns found.")
    else:
        for name, count in checks:
            print(f"{name}: {count:,}")

    print_section("7. NUMERIC FEATURE SUMMARY")

    numeric_columns = [
        "vehicle_age_years",
        "latest_mileage",
        "average_annual_mileage",
        "mileage_since_previous_mot",
        "mot_event_count",
        "historical_failure_rate",
        "failures_last_3_events",
        "historical_major_defects_last_3",
        "historical_dangerous_defects_last_3",
        "latest_advisory_count",
        "days_since_last_mot",
    ]

    available_numeric = [
        column
        for column in numeric_columns
        if column in df.columns
    ]

    if available_numeric:
        numeric_df = (
            df[available_numeric]
            .apply(pd.to_numeric, errors="coerce")
        )

        summary = numeric_df.describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        ).transpose()

        print(summary.to_string())

    print_section("8. CATEGORICAL DISTRIBUTIONS")

    for column in [
        "make",
        "model",
        "fuel_type",
    ]:
        if column not in df.columns:
            continue

        print()
        print(f"Top 15 values for {column}:")

        print(
            df[column]
            .fillna("<MISSING>")
            .astype(str)
            .value_counts()
            .head(15)
            .to_string()
        )

    print_section("9. IMPORTANT MODEL-SPLIT WARNING")

    print(
        "Do NOT use train_test_split() directly on rows."
    )
    print(
        "A single registration usually contributes multiple "
        "historical prediction rows."
    )
    print(
        "If rows from the same vehicle appear in both training "
        "and testing data, evaluation will be overly optimistic."
    )
    print()
    print(
        "For the first model we will split by registration "
        "(vehicle-level group split)."
    )
    print(
        "For the dissertation we should additionally evaluate "
        "a chronological/time-based holdout."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the MOT training dataset before model training."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/mot_training_sample.csv",
        help=(
            "Training CSV path. "
            "Default: data/processed/mot_training_sample.csv"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    inspect_dataset(Path(args.input))


if __name__ == "__main__":
    main()