from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd

from app.services.feature_engineering import (
    DEFECT_CATEGORIES,
    build_mot_events,
    calculate_vehicle_age,
    categorise_defect,
    extract_mot_features,
    get_initial_defects,
    parse_datetime,
)


TARGET_CATEGORIES = list(DEFECT_CATEGORIES.keys()) + ["other"]


def _vehicle_with_history_tests(
    vehicle_data: dict,
    history_events: list[dict],
) -> dict:
    """
    Create a copy of a DVSA vehicle record containing ONLY tests
    that occurred before the target MOT event.

    This is the core protection against target leakage.
    """

    history_tests = []

    for event in history_events:
        history_tests.extend(
            event.get("tests", [])
        )

    history_vehicle = {
        key: value
        for key, value in vehicle_data.items()
        if key != "motTests"
    }

    history_vehicle["motTests"] = history_tests

    return history_vehicle


def _target_failure_categories(
    target_event: dict,
) -> dict[str, int]:
    """
    Create multi-label targets showing which component categories
    caused a MAJOR or DANGEROUS defect on the target event's
    initial test.

    These columns are targets only; they must never be used as
    predictor inputs.
    """

    category_counts = Counter()

    for defect in get_initial_defects(
        target_event
    ):
        defect_type = defect.get(
            "type",
            ""
        ).upper()

        if defect_type not in {
            "MAJOR",
            "DANGEROUS",
        }:
            continue

        for category in categorise_defect(
            defect.get(
                "text",
                ""
            )
        ):
            category_counts[category] += 1

    return {
        f"target_failure_{category}":
            int(
                category_counts.get(
                    category,
                    0,
                ) > 0
            )
        for category in TARGET_CATEGORIES
    }


def build_training_rows_for_vehicle(
    vehicle_data: dict,
    min_history_events: int = 1,
) -> list[dict]:
    """
    Turn one vehicle's MOT history into multiple chronological
    training examples.

    Example:

        2019 history -> predict 2020
        2019-2020 history -> predict 2021
        2019-2021 history -> predict 2022
        ...

    IMPORTANT:
    For every row, the feature extractor sees ONLY events that
    occurred before the target event.
    """

    events = build_mot_events(
        vehicle_data
    )

    if len(events) <= min_history_events:
        return []

    rows = []

    for target_index in range(
        min_history_events,
        len(events),
    ):
        history_events = events[
            :target_index
        ]

        target_event = events[
            target_index
        ]

        target_result = (
            target_event.get(
                "initial_result"
            )
        )

        # Ignore unusual/unsupported outcome values.
        if target_result not in {
            "PASSED",
            "FAILED",
        }:
            continue

        target_date = parse_datetime(
            target_event.get(
                "start_date"
            )
        )

        if target_date is None:
            continue

        history_vehicle = (
            _vehicle_with_history_tests(
                vehicle_data,
                history_events,
            )
        )

        # All normal engineered features are calculated using
        # ONLY the historical tests.
        features = extract_mot_features(
            history_vehicle
        )

        # extract_mot_features() is also used for live analysis,
        # where vehicle age is calculated using today's date.
        #
        # For historical training rows we MUST recalculate age at
        # the date of the target MOT, otherwise old examples would
        # incorrectly use the vehicle's present-day age.
        features[
            "vehicle_age_years"
        ] = calculate_vehicle_age(
            vehicle_data.get(
                "firstUsedDate"
            ),
            as_of=target_date,
        )

        previous_event = history_events[
            -1
        ]

        previous_event_date = (
            parse_datetime(
                previous_event.get(
                    "end_date"
                )
            )
        )

        days_since_last_mot = None

        if previous_event_date is not None:
            difference = (
                target_date
                - previous_event_date
            ).days

            if difference >= 0:
                days_since_last_mot = (
                    difference
                )

        features[
            "days_since_last_mot"
        ] = days_since_last_mot

        row = {
            # Metadata.
            # Do not feed these directly into the first ML model.
            "registration":
                vehicle_data.get(
                    "registration"
                ),

            "make":
                vehicle_data.get(
                    "make"
                ),

            "model":
                vehicle_data.get(
                    "model"
                ),

            "fuel_type":
                vehicle_data.get(
                    "fuelType"
                ),

            "target_date":
                target_event.get(
                    "start_date"
                ),

            # Number of MOT events the model was allowed to see.
            "history_event_count":
                len(
                    history_events
                ),

            **features,

            # Binary target:
            # 0 = pass on initial test
            # 1 = fail on initial test
            "target_failed":
                int(
                    target_result
                    == "FAILED"
                ),

            "target_result":
                target_result,

            # Additional targets for the later defect-category model.
            **_target_failure_categories(
                target_event
            ),
        }

        rows.append(row)

    return rows


def build_training_rows(
    vehicles: Iterable[dict],
    min_history_events: int = 1,
) -> list[dict]:
    """
    Generate training rows for many vehicle records.
    """

    rows = []

    for vehicle_data in vehicles:
        rows.extend(
            build_training_rows_for_vehicle(
                vehicle_data,
                min_history_events=(
                    min_history_events
                ),
            )
        )

    return rows


def build_training_dataframe(
    vehicles: Iterable[dict],
    min_history_events: int = 1,
) -> pd.DataFrame:
    """
    Generate a pandas DataFrame for model training.
    """

    rows = build_training_rows(
        vehicles,
        min_history_events=(
            min_history_events
        ),
    )

    return pd.DataFrame(
        rows
    )


def save_training_csv(
    vehicles: Iterable[dict],
    output_path: str,
    min_history_events: int = 1,
) -> pd.DataFrame:
    """
    Build the training dataset and save it to CSV.

    Returns the DataFrame as well so callers can inspect it.
    """

    dataframe = (
        build_training_dataframe(
            vehicles,
            min_history_events=(
                min_history_events
            ),
        )
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return dataframe