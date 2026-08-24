from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.services.feature_engineering import (
    build_mot_events,
    extract_mot_features,
    get_final_test,
    parse_datetime,
)


PASS_FAIL_MODEL_PATH = Path(
    "models/mot_pass_fail_model.joblib"
)

PASS_FAIL_FEATURE_CONFIG_PATH = Path(
    "models/model_features.json"
)

PASS_FAIL_THRESHOLD_CONFIG_PATH = Path(
    "reports/decision_threshold.json"
)

CATEGORY_MODEL_PATH = Path(
    "models/mot_failure_category_model.joblib"
)

CATEGORY_FEATURE_CONFIG_PATH = Path(
    "models/category_model_features.json"
)

CATEGORY_THRESHOLD_CONFIG_PATH = Path(
    "reports/category_thresholds.json"
)


CATEGORY_LABELS = {
    "tyres":
        "Tyres",

    "brakes":
        "Brakes",

    "suspension":
        "Suspension",

    "lighting":
        "Lighting",

    "steering":
        "Steering",

    "visibility":
        "Visibility",

    "emissions_exhaust":
        "Emissions / exhaust",

    "body_structure":
        "Body / structure",

    "seatbelts_srs":
        "Seatbelts / SRS",

    "other":
        "Other",
}


@lru_cache(maxsize=1)
def load_pass_fail_model():
    if not PASS_FAIL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: "
            f"{PASS_FAIL_MODEL_PATH}"
        )

    return joblib.load(
        PASS_FAIL_MODEL_PATH
    )


@lru_cache(maxsize=1)
def load_category_model():
    if not CATEGORY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Category model not found: "
            f"{CATEGORY_MODEL_PATH}"
        )

    return joblib.load(
        CATEGORY_MODEL_PATH
    )


def _load_json(
    path: Path,
    description: str,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_pass_fail_feature_config() -> dict:
    return _load_json(
        PASS_FAIL_FEATURE_CONFIG_PATH,
        "PASS/FAIL feature config",
    )


@lru_cache(maxsize=1)
def load_pass_fail_threshold_config() -> dict:
    return _load_json(
        PASS_FAIL_THRESHOLD_CONFIG_PATH,
        "PASS/FAIL threshold config",
    )


@lru_cache(maxsize=1)
def load_category_feature_config() -> dict:
    return _load_json(
        CATEGORY_FEATURE_CONFIG_PATH,
        "Category feature config",
    )


@lru_cache(maxsize=1)
def load_category_threshold_config() -> dict:
    return _load_json(
        CATEGORY_THRESHOLD_CONFIG_PATH,
        "Category threshold config",
    )


def _to_date(
    value: str | None,
) -> date | None:
    parsed = parse_datetime(
        value
    )

    if parsed is None:
        return None

    return parsed.date()


def calculate_historical_mot_intervals(
    vehicle_data: dict,
) -> list[int]:
    """
    Historical interval:
        previous MOT event end date
        ->
        next MOT event start date

    This matches the training meaning of
    `days_since_last_mot`.
    """

    events = build_mot_events(
        vehicle_data
    )

    intervals = []

    for index in range(
        1,
        len(events),
    ):
        previous_date = _to_date(
            events[
                index - 1
            ].get(
                "end_date"
            )
        )

        next_date = _to_date(
            events[
                index
            ].get(
                "start_date"
            )
        )

        if (
            previous_date is None
            or next_date is None
        ):
            continue

        days = (
            next_date
            - previous_date
        ).days

        if days >= 0:
            intervals.append(
                days
            )

    return intervals


def calculate_prediction_horizon(
    vehicle_data: dict,
) -> dict:
    """
    Estimate the interval from the latest MOT to the next MOT.

    Preferred:
        latest final test expiryDate

    Fallback:
        median historical MOT interval

    Final fallback:
        365 days
    """

    events = build_mot_events(
        vehicle_data
    )

    if not events:
        return {
            "days": None,
            "source": "unavailable",
            "latest_mot_date": None,
            "mot_expiry_date": None,
        }

    latest_event = events[
        -1
    ]

    latest_mot_date = _to_date(
        latest_event.get(
            "end_date"
        )
    )

    final_test = get_final_test(
        latest_event
    )

    expiry_date = _to_date(
        final_test.get(
            "expiryDate"
        )
    )

    if (
        latest_mot_date is not None
        and expiry_date is not None
    ):
        days = (
            expiry_date
            - latest_mot_date
        ).days

        if days >= 0:
            return {
                "days":
                    days,

                "source":
                    "latest_mot_expiry",

                "latest_mot_date":
                    latest_mot_date.isoformat(),

                "mot_expiry_date":
                    expiry_date.isoformat(),
            }

    intervals = (
        calculate_historical_mot_intervals(
            vehicle_data
        )
    )

    if intervals:
        estimated_days = int(
            round(
                median(
                    intervals
                )
            )
        )

        return {
            "days":
                estimated_days,

            "source":
                "vehicle_historical_median",

            "latest_mot_date": (
                latest_mot_date.isoformat()
                if latest_mot_date
                else None
            ),

            "mot_expiry_date":
                None,
        }

    return {
        "days":
            365,

        "source":
            "default_365_days",

        "latest_mot_date": (
            latest_mot_date.isoformat()
            if latest_mot_date
            else None
        ),

        "mot_expiry_date":
            None,
    }


def build_live_engineered_features(
    vehicle_data: dict,
) -> tuple[
    dict[str, Any],
    dict,
]:
    """
    Build the shared live feature dictionary used by both models.
    """

    engineered = extract_mot_features(
        vehicle_data
    )

    horizon = (
        calculate_prediction_horizon(
            vehicle_data
        )
    )

    engineered[
        "days_since_last_mot"
    ] = horizon[
        "days"
    ]

    engineered[
        "make"
    ] = vehicle_data.get(
        "make"
    )

    engineered[
        "model"
    ] = vehicle_data.get(
        "model"
    )

    engineered[
        "fuel_type"
    ] = vehicle_data.get(
        "fuelType"
    )

    return (
        engineered,
        horizon,
    )


def select_expected_features(
    engineered: dict[str, Any],
    config: dict,
) -> dict[str, Any]:
    expected_columns = (
        config.get(
            "feature_columns",
            []
        )
    )

    if not expected_columns:
        raise RuntimeError(
            "Model feature configuration "
            "does not contain feature_columns."
        )

    return {
        column:
            engineered.get(
                column
            )
        for column in expected_columns
    }


def get_pass_fail_warning_threshold() -> float:
    config = (
        load_pass_fail_threshold_config()
    )

    warning = config.get(
        "high_recall_warning",
        {},
    )

    return float(
        warning.get(
            "threshold",
            0.50,
        )
    )


def probability_risk_band(
    failure_probability: float,
) -> str:
    """
    Presentation-only bands.

    These are not legal MOT thresholds.
    """

    if failure_probability < 0.20:
        return "LOW"

    if failure_probability < 0.35:
        return "MODERATE"

    if failure_probability < 0.50:
        return "ELEVATED"

    return "HIGH"


def predict_pass_fail(
    engineered: dict[str, Any],
) -> dict:
    model = load_pass_fail_model()

    feature_config = (
        load_pass_fail_feature_config()
    )

    features = (
        select_expected_features(
            engineered,
            feature_config,
        )
    )

    frame = pd.DataFrame(
        [features]
    )

    probabilities = (
        model.predict_proba(
            frame
        )[0]
    )

    classifier = (
        model.named_steps.get(
            "classifier"
        )
    )

    if classifier is None:
        raise RuntimeError(
            "PASS/FAIL pipeline does not "
            "contain a classifier step."
        )

    classes = list(
        classifier.classes_
    )

    if 1 not in classes:
        raise RuntimeError(
            "PASS/FAIL classifier does not "
            "contain failure class 1."
        )

    fail_index = (
        classes.index(
            1
        )
    )

    failure_probability = float(
        probabilities[
            fail_index
        ]
    )

    pass_probability = (
        1.0
        - failure_probability
    )

    threshold = (
        get_pass_fail_warning_threshold()
    )

    warning_triggered = (
        failure_probability
        >= threshold
    )

    return {
        "pass_probability":
            round(
                pass_probability,
                4,
            ),

        "fail_probability":
            round(
                failure_probability,
                4,
            ),

        "pass_probability_percent":
            round(
                pass_probability
                * 100,
                1,
            ),

        "fail_probability_percent":
            round(
                failure_probability
                * 100,
                1,
            ),

        "risk_band":
            probability_risk_band(
                failure_probability
            ),

        "warning_threshold":
            round(
                threshold,
                4,
            ),

        "warning_triggered":
            warning_triggered,

        "assessment": (
            "INCREASED_FAILURE_RISK"
            if warning_triggered
            else "LOWER_FAILURE_RISK"
        ),
    }


def predict_failure_categories(
    engineered: dict[str, Any],
    overall_warning_triggered: bool,
) -> dict:
    """
    Predict category risk scores conditional on MOT failure.

    IMPORTANT:
    The category LogisticRegression models used
    class_weight='balanced', so their raw predict_proba outputs are
    treated as MODEL SCORES, not calibrated real-world probabilities.

    A category becomes an inspection priority only when:
        score >= validation-selected category threshold

    If overall PASS/FAIL warning is not triggered, categories are
    still calculated internally but are not surfaced as primary
    inspection priorities.
    """

    model = load_category_model()

    feature_config = (
        load_category_feature_config()
    )

    threshold_config = (
        load_category_threshold_config()
    )

    categories = feature_config.get(
        "categories",
        []
    )

    if not categories:
        raise RuntimeError(
            "Category feature configuration "
            "does not contain categories."
        )

    features = (
        select_expected_features(
            engineered,
            feature_config,
        )
    )

    frame = pd.DataFrame(
        [features]
    )

    scores = np.asarray(
        model.predict_proba(
            frame
        )
    )

    if scores.ndim != 2:
        raise RuntimeError(
            "Unexpected category model output shape: "
            f"{scores.shape}"
        )

    if scores.shape[
        1
    ] != len(
        categories
    ):
        raise RuntimeError(
            "Category model output count does not "
            "match configured categories."
        )

    category_thresholds = (
        threshold_config.get(
            "categories",
            {}
        )
    )

    all_scores = []

    flagged = []

    for index, category in enumerate(
        categories
    ):
        score = float(
            scores[
                0,
                index
            ]
        )

        category_config = (
            category_thresholds.get(
                category,
                {}
            )
        )

        threshold = float(
            category_config.get(
                "warning_threshold",
                0.50,
            )
        )

        is_flagged = (
            score
            >= threshold
        )

        margin = (
            score
            - threshold
        )

        result = {
            "category":
                category,

            "label":
                CATEGORY_LABELS.get(
                    category,
                    category,
                ),

            "model_score":
                round(
                    score,
                    4,
                ),

            "warning_threshold":
                round(
                    threshold,
                    4,
                ),

            "threshold_margin":
                round(
                    margin,
                    4,
                ),

            "flagged":
                is_flagged,
        }

        all_scores.append(
            result
        )

        if is_flagged:
            flagged.append(
                result
            )

    # Scores are not calibrated across categories, so sorting by
    # threshold margin is more defensible than calling the highest
    # raw score the highest probability.
    flagged.sort(
        key=lambda item:
            item[
                "threshold_margin"
            ],
        reverse=True,
    )

    if overall_warning_triggered:
        inspection_priorities = (
            flagged[
                :5
            ]
        )
    else:
        inspection_priorities = []

    return {
        "inspection_priorities":
            inspection_priorities,

        "number_of_flagged_categories":
            len(
                flagged
            ),

        "category_scores":
            all_scores,

        "category_model_note": (
            "Category values are model scores used for ranking "
            "and threshold decisions, not calibrated probabilities. "
            "They estimate likely failure areas conditional on an "
            "MOT failure. Categories are surfaced only when their "
            "validation-selected warning threshold is crossed."
        ),
    }


def predict_mot_risk(
    vehicle_data: dict,
) -> dict:
    """
    Complete live MOT risk prediction.

    Produces:
        1. PASS/FAIL probability
        2. validation-selected warning decision
        3. category inspection priorities
    """

    (
        engineered,
        prediction_context,
    ) = build_live_engineered_features(
        vehicle_data
    )

    pass_fail = (
        predict_pass_fail(
            engineered
        )
    )

    categories = (
        predict_failure_categories(
            engineered,
            overall_warning_triggered=(
                pass_fail[
                    "warning_triggered"
                ]
            ),
        )
    )

    return {
        **pass_fail,

        "prediction_context": {
            "prediction_target":
                "NEXT_MOT_INITIAL_RESULT",

            "mot_interval_days_used":
                prediction_context.get(
                    "days"
                ),

            "mot_interval_source":
                prediction_context.get(
                    "source"
                ),

            "latest_mot_date":
                prediction_context.get(
                    "latest_mot_date"
                ),

            "mot_expiry_date":
                prediction_context.get(
                    "mot_expiry_date"
                ),
        },

        "failure_categories":
            categories,

        "model_note": (
            "This is a statistical estimate of the risk that the "
            "vehicle's next MOT initial test will fail, based on "
            "historical MOT and vehicle data. Failure-category "
            "outputs are inspection-priority scores rather than "
            "calibrated probabilities. The system does not replace "
            "a physical MOT inspection and does not guarantee a "
            "pass or fail result."
        ),
    }