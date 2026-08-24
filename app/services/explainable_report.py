from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.services.feature_engineering import (
    build_mot_events,
    categorise_defect,
    extract_mot_features,
    get_final_defects,
)
from app.services.predictor import predict_mot_risk


CATEGORY_LABELS = {
    "tyres": "Tyres",
    "brakes": "Brakes",
    "suspension": "Suspension",
    "lighting": "Lighting",
    "steering": "Steering",
    "visibility": "Visibility",
    "emissions_exhaust": "Emissions / exhaust",
    "body_structure": "Body / structure",
    "seatbelts_srs": "Seatbelts / SRS",
    "other": "Other",
}


def _latest_advisory_evidence(
    vehicle_data: dict,
) -> dict[str, dict]:
    """
    Collect advisory defects that remain on the latest FINAL MOT result.
    """

    events = build_mot_events(vehicle_data)

    evidence = defaultdict(
        lambda: {
            "count": 0,
            "texts": [],
        }
    )

    if not events:
        return {}

    latest_event = events[-1]

    for defect in get_final_defects(latest_event):
        if str(defect.get("type", "")).upper() != "ADVISORY":
            continue

        text = str(defect.get("text", "")).strip()

        for category in categorise_defect(text):
            evidence[category]["count"] += 1

            if (
                text
                and text not in evidence[category]["texts"]
            ):
                evidence[category]["texts"].append(text)

    return dict(evidence)


def _category_history_evidence(
    features: dict,
    latest_advisories: dict,
    category: str,
) -> dict[str, Any]:
    recent_advisories = int(
        features.get(
            f"{category}_advisories_last_2",
            0,
        )
        or 0
    )

    repeated_advisory = bool(
        features.get(
            f"repeated_{category}_advisory",
            0,
        )
    )

    historical_failures = int(
        features.get(
            f"historical_{category}_failures_last_3",
            0,
        )
        or 0
    )

    latest = latest_advisories.get(
        category,
        {},
    )

    latest_advisory_count = int(
        latest.get("count", 0)
        or 0
    )

    latest_advisory_texts = list(
        latest.get("texts", [])
    )

    has_history_support = any(
        [
            recent_advisories > 0,
            repeated_advisory,
            historical_failures > 0,
            latest_advisory_count > 0,
        ]
    )

    # "Strong" support is deliberately stricter than simple history support.
    #
    # A category becomes a likely future failure area when there has been:
    #   - at least one serious historical failure in the last 3 events, OR
    #   - the same advisory category across both of the last 2 final MOTs.
    #
    # This prevents a single low-severity advisory from being presented with
    # the same importance as repeated or serious historical defects.
    strong_history_support = (
        historical_failures > 0
        or repeated_advisory
    )

    evidence_score = (
        recent_advisories
        + (
            3
            if repeated_advisory
            else 0
        )
        + (
            historical_failures * 2
        )
        + (
            latest_advisory_count * 2
        )
    )

    reasons = []

    if historical_failures > 0:
        reasons.append(
            (
                f"{historical_failures} serious "
                f"{CATEGORY_LABELS.get(category, category).lower()} "
                "defect(s) occurred across the last three MOT events."
            )
        )

    if repeated_advisory:
        reasons.append(
            (
                f"{CATEGORY_LABELS.get(category, category)} advisories "
                "appeared in both of the two most recent final MOT results."
            )
        )

    if recent_advisories > 0:
        reasons.append(
            (
                f"{recent_advisories} advisory item(s) in this category "
                "appeared across the two most recent final MOT results."
            )
        )

    if latest_advisory_count > 0:
        reasons.append(
            (
                f"{latest_advisory_count} advisory item(s) in this category "
                "remain on the latest MOT result."
            )
        )

    return {
        "recent_advisories_last_2":
            recent_advisories,

        "repeated_advisory":
            repeated_advisory,

        "historical_serious_failures_last_3":
            historical_failures,

        "latest_advisory_count":
            latest_advisory_count,

        "latest_advisory_texts":
            latest_advisory_texts,

        "history_support":
            has_history_support,

        "strong_history_support":
            strong_history_support,

        "evidence_score":
            evidence_score,

        "reasons":
            reasons,
    }


def _inspection_action(
    category: str,
) -> str:
    actions = {
        "tyres":
            (
                "Check tread depth, damage, uneven wear and tyre condition "
                "on all wheels."
            ),

        "brakes":
            (
                "Have brake pipes, pads and discs inspected, especially "
                "where corrosion or wear has appeared previously."
            ),

        "suspension":
            (
                "Inspect suspension joints, links, springs and shock "
                "absorbers for play, damage or leakage."
            ),

        "lighting":
            (
                "Check all required lamps, indicators and fog lights for "
                "operation, condition and security."
            ),

        "steering":
            (
                "Inspect steering joints and components for excessive play, "
                "wear or damage."
            ),

        "visibility":
            (
                "Check windscreen condition, wipers, washers and mirrors."
            ),

        "emissions_exhaust":
            (
                "Inspect the exhaust system for damage, leaks and insecure "
                "components."
            ),

        "body_structure":
            (
                "Inspect body and structural areas for corrosion, damage or "
                "insecure components."
            ),

        "seatbelts_srs":
            (
                "Check seatbelts, warning lamps and restraint-system "
                "operation."
            ),

        "other":
            (
                "Review the latest MOT advisories and arrange a general "
                "pre-MOT inspection."
            ),
    }

    return actions.get(
        category,
        "Arrange a pre-MOT inspection of this area.",
    )


def _build_category_item(
    category_result: dict,
    evidence: dict,
) -> dict:
    category = category_result["category"]

    return {
        "category":
            category,

        "label":
            CATEGORY_LABELS.get(
                category,
                category,
            ),

        "model_score":
            category_result["model_score"],

        "warning_threshold":
            category_result["warning_threshold"],

        "threshold_margin":
            category_result["threshold_margin"],

        "model_flagged":
            category_result["flagged"],

        "history_support":
            evidence["history_support"],

        "strong_history_support":
            evidence["strong_history_support"],

        "evidence_score":
            evidence["evidence_score"],

        "evidence":
            evidence,

        "recommended_check":
            _inspection_action(category),
    }


def _overall_summary(
    prediction: dict,
    likely_failure_count: int,
    current_advisory_count: int,
) -> str:
    fail_percent = prediction[
        "fail_probability_percent"
    ]

    risk_band = prediction[
        "risk_band"
    ]

    if prediction["warning_triggered"]:
        if likely_failure_count:
            return (
                f"The model estimates a {fail_percent:.1f}% risk of initial "
                f"MOT failure ({risk_band} risk). The strongest combined "
                "model-and-history evidence identifies "
                f"{likely_failure_count} likely failure area(s) for "
                "pre-MOT attention."
            )

        if current_advisory_count:
            return (
                f"The model estimates a {fail_percent:.1f}% risk of initial "
                f"MOT failure ({risk_band} risk). No category meets the "
                "stricter likely-failure rule, but current MOT advisories "
                "should still be checked before the next test."
            )

        return (
            f"The model estimates a {fail_percent:.1f}% risk of initial "
            f"MOT failure ({risk_band} risk), although no category currently "
            "has strong corroborating MOT-history evidence."
        )

    return (
        f"The model estimates a {fail_percent:.1f}% risk of initial MOT "
        f"failure ({risk_band} risk), below the current warning threshold."
    )


def build_explainable_mot_report(
    vehicle_data: dict,
    max_likely_failure_areas: int = 5,
    max_current_advisory_checks: int = 5,
    max_watchlist_items: int = 3,
) -> dict:
    """
    Build the final user-facing explainable MOT-readiness report.

    The report separates three concepts:

    1. likely_failure_areas
       ML threshold crossed + STRONG historical support
       (serious historical failure and/or repeated advisory).

    2. current_advisory_checks
       Advisory categories still present on the latest final MOT result,
       excluding categories already promoted above.

    3. watchlist
       ML threshold crossed + some historical support, but not enough to
       qualify as a likely failure area.

    Raw model-only signals remain available for development transparency but
    should not be shown prominently to normal users.
    """

    prediction = predict_mot_risk(
        vehicle_data
    )

    features = extract_mot_features(
        vehicle_data
    )

    latest_advisories = _latest_advisory_evidence(
        vehicle_data
    )

    raw_categories = (
        prediction
        .get(
            "failure_categories",
            {},
        )
        .get(
            "category_scores",
            [],
        )
    )

    likely_failure_areas = []
    current_advisory_candidates = []
    watchlist = []
    model_only = []

    for category_result in raw_categories:
        category = category_result[
            "category"
        ]

        evidence = _category_history_evidence(
            features,
            latest_advisories,
            category,
        )

        item = _build_category_item(
            category_result,
            evidence,
        )

        if (
            item["model_flagged"]
            and item[
                "strong_history_support"
            ]
        ):
            likely_failure_areas.append(
                item
            )

        elif (
            item["model_flagged"]
            and item["history_support"]
        ):
            watchlist.append(
                item
            )

        elif item["model_flagged"]:
            model_only.append(
                item
            )

        # Latest advisories are useful even if the category model is weak.
        if (
            evidence[
                "latest_advisory_count"
            ] > 0
        ):
            current_advisory_candidates.append(
                item
            )

    likely_failure_areas.sort(
        key=lambda item: (
            item["evidence_score"],
            item["threshold_margin"],
        ),
        reverse=True,
    )

    likely_failure_areas = (
        likely_failure_areas[
            :max_likely_failure_areas
        ]
    )

    promoted_categories = {
        item["category"]
        for item in likely_failure_areas
    }

    # Avoid repeating the same category in both major sections.
    current_advisory_checks = [
        item
        for item in current_advisory_candidates
        if item[
            "category"
        ] not in promoted_categories
    ]

    current_advisory_checks.sort(
        key=lambda item: (
            item[
                "evidence"
            ][
                "latest_advisory_count"
            ],
            item["threshold_margin"],
        ),
        reverse=True,
    )

    current_advisory_checks = (
        current_advisory_checks[
            :max_current_advisory_checks
        ]
    )

    current_advisory_categories = {
        item["category"]
        for item in current_advisory_checks
    }

    # Keep watchlist distinct from the two more actionable sections.
    watchlist = [
        item
        for item in watchlist
        if item["category"]
        not in promoted_categories
        and item["category"]
        not in current_advisory_categories
    ]

    watchlist.sort(
        key=lambda item: (
            item["evidence_score"],
            item["threshold_margin"],
        ),
        reverse=True,
    )

    watchlist = watchlist[
        :max_watchlist_items
    ]

    model_only.sort(
        key=lambda item:
            item["threshold_margin"],
        reverse=True,
    )

    summary = _overall_summary(
        prediction,
        likely_failure_count=len(
            likely_failure_areas
        ),
        current_advisory_count=len(
            current_advisory_checks
        ),
    )

    return {
        "vehicle": {
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

            "first_used_date":
                vehicle_data.get(
                    "firstUsedDate"
                ),
        },

        "readiness": {
            "pass_probability_percent":
                prediction[
                    "pass_probability_percent"
                ],

            "fail_probability_percent":
                prediction[
                    "fail_probability_percent"
                ],

            "risk_band":
                prediction[
                    "risk_band"
                ],

            "warning_threshold":
                prediction[
                    "warning_threshold"
                ],

            "warning_triggered":
                prediction[
                    "warning_triggered"
                ],

            "assessment":
                prediction[
                    "assessment"
                ],
        },

        "summary":
            summary,

        "likely_failure_areas":
            likely_failure_areas,

        "current_advisory_checks":
            current_advisory_checks,

        "watchlist":
            watchlist,

        "development_signals": {
            "model_only_count":
                len(
                    model_only
                ),

            "top_model_only_signals":
                model_only[
                    :3
                ],

            "note":
                (
                    "These model signals were not promoted because the "
                    "vehicle's recent MOT history did not independently "
                    "support them. They are intended for development and "
                    "diagnostic transparency rather than primary user advice."
                ),
        },

        "history_snapshot": {
            "mot_event_count":
                features.get(
                    "mot_event_count"
                ),

            "historical_failure_rate":
                features.get(
                    "historical_failure_rate"
                ),

            "failures_last_3_events":
                features.get(
                    "failures_last_3_events"
                ),

            "latest_advisory_count":
                features.get(
                    "latest_advisory_count"
                ),

            "latest_major_count":
                features.get(
                    "latest_major_count"
                ),

            "latest_dangerous_count":
                features.get(
                    "latest_dangerous_count"
                ),
        },

        "prediction_context":
            prediction[
                "prediction_context"
            ],

        "important_note": (
            "This is an explainable statistical MOT-readiness assessment. "
            "Likely failure areas require both an ML warning signal and "
            "strong historical evidence. Current advisory checks are shown "
            "separately because an advisory is not the same as a predicted "
            "future MOT failure. The report does not replace a physical "
            "inspection and does not guarantee the result of a future MOT."
        ),
    }