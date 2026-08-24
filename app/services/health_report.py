RISK_CATEGORIES = [
    "tyres",
    "brakes",
    "suspension",
    "lighting",
    "steering",
    "visibility",
    "emissions_exhaust",
    "body_structure",
    "seatbelts_srs",
]


def calculate_category_risk(
    features: dict,
    category: str,
) -> dict:

    advisories = features.get(
        f"{category}_advisories_last_2",
        0,
    )

    major = features.get(
        f"{category}_major_last_2",
        0,
    )

    dangerous = features.get(
        f"{category}_dangerous_last_2",
        0,
    )

    repeated = features.get(
        f"repeated_{category}_advisory",
        0,
    )

    # Temporary rule-based score.
    #
    # IMPORTANT:
    # This is NOT our machine-learning prediction.
    # We will replace it later.

    risk_points = (
        advisories
        + (major * 3)
        + (dangerous * 5)
        + (repeated * 2)
    )

    if risk_points >= 7:
        level = "HIGH"

    elif risk_points >= 3:
        level = "MEDIUM"

    else:
        level = "LOW"

    return {
        "category": category,
        "risk_level": level,
        "risk_points": risk_points,
        "recent_advisories": advisories,
        "recent_major": major,
        "recent_dangerous": dangerous,
        "repeated_advisory": bool(
            repeated
        ),
    }


def build_health_report(
    features: dict,
) -> dict:

    category_risks = [
        calculate_category_risk(
            features,
            category,
        )
        for category in RISK_CATEGORIES
    ]

    category_risks.sort(
        key=lambda item: item[
            "risk_points"
        ],
        reverse=True,
    )

    high_risk = [
        item
        for item in category_risks
        if item["risk_level"] == "HIGH"
    ]

    medium_risk = [
        item
        for item in category_risks
        if item["risk_level"] == "MEDIUM"
    ]

    if high_risk:
        overall_risk = "HIGH"

    elif (
        medium_risk
        or features.get(
            "failures_last_3_events",
            0,
        ) > 0
    ):
        overall_risk = "MEDIUM"

    else:
        overall_risk = "LOW"

    return {
        "overall_risk": overall_risk,

        "recent_failures":
            features.get(
                "failures_last_3_events",
                0,
            ),

        "risk_categories":
            category_risks,

        "important_note": (
            "This is a prototype rule-based "
            "risk assessment, not the final "
            "machine-learning MOT prediction."
        ),
    }