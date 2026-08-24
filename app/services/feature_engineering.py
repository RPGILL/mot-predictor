from collections import Counter
from datetime import datetime
from typing import Optional
import re


# ---------------------------------------------------------------------------
# MOT RETEST GROUPING SETTINGS
# ---------------------------------------------------------------------------

# Historical MOT data does not explicitly tell us that two tests belong to the
# same retest episode. These values are therefore prototype heuristics.
#
# A test is treated as a probable retest when:
#   1. the previous MOT event initially failed,
#   2. the new test occurs within 14 days, and
#   3. the normalised mileage difference is no more than 500 miles.
#
# These values can later be evaluated and adjusted during the dissertation.
RETEST_WINDOW_DAYS = 14
RETEST_MAX_MILES = 500


# ---------------------------------------------------------------------------
# DEFECT CATEGORIES
# ---------------------------------------------------------------------------

DEFECT_CATEGORIES = {
    "tyres": [
        "tyre",
        "tire",
    ],

    "brakes": [
        "brake",
    ],

    "suspension": [
        "suspension",
        "shock absorber",
        "shock absorbers",
        "anti-roll",
        "coil spring",
        "ball joint",
    ],

    "lighting": [
        "lamp",
        "light",
        "headlamp",
        "indicator",
        "fog light",
    ],

    "steering": [
        "steering",
        "track rod",
    ],

    "visibility": [
        "windscreen",
        "wiper",
        "washer",
        "mirror",
    ],

    "emissions_exhaust": [
        "emission",
        "exhaust",
        "catalytic",
        "lambda",
        "smoke",
    ],

    "body_structure": [
        "bodywork",
        "body",
        "structure",
        "chassis",
        "subframe",
        "sub-frame",
        "sill",
        "floor",
        "crossmember",
        "cross member",
    ],

    "seatbelts_srs": [
        "seat belt",
        "seatbelt",
        "airbag",
        "supplementary restraint",
        "srs",
    ],
}


# ---------------------------------------------------------------------------
# BASIC HELPERS
# ---------------------------------------------------------------------------

def contains_keyword(text: str, keyword: str) -> bool:
    """
    Match a complete word or phrase.

    This prevents false substring matches such as:
        "light" matching "slight"
        "lamp" matching "clamp"
    """

    text = (text or "").lower()
    keyword = (keyword or "").lower()

    pattern = (
        r"(?<!\w)"
        + re.escape(keyword)
        + r"(?!\w)"
    )

    return re.search(pattern, text) is not None


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """
    Convert a DVSA ISO date/datetime string into a Python datetime.
    """

    if not value:
        return None

    try:
        value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)

    except (ValueError, TypeError):
        return None


def safe_int(value) -> Optional[int]:
    """
    Convert a value to int safely.
    """

    if value is None:
        return None

    try:
        return int(value)

    except (ValueError, TypeError):
        return None




def get_test_mileage_miles(test: dict) -> Optional[int]:
    """
    Return a DVSA odometer reading normalised to miles.

    DVSA records can use MI or KM. Readings reported as
    NO_ODOMETER (or with an unknown unit) are treated as missing.
    """

    result_type = str(
        test.get("odometerResultType", "")
    ).upper()

    if result_type == "NO_ODOMETER":
        return None

    value = safe_int(
        test.get("odometerValue")
    )

    if value is None:
        return None

    unit = str(
        test.get("odometerUnit", "")
    ).upper()

    if unit == "MI":
        return value

    if unit == "KM":
        return round(
            value / 1.609344
        )

    # Unknown/missing units are safer to treat as missing than
    # to mix incompatible measurements in the ML features.
    return None


def normalise_defect_text(text: str) -> str:
    """
    Normalise whitespace and casing so duplicate defect descriptions
    can be compared consistently.
    """

    return " ".join((text or "").lower().split())


def categorise_defect(text: str) -> list[str]:
    """
    Map a DVSA defect description into one or more component categories.

    A defect may belong to more than one category if its wording genuinely
    references multiple vehicle systems.
    """

    categories = []

    for category, keywords in DEFECT_CATEGORIES.items():

        matched = any(
            contains_keyword(text, keyword)
            for keyword in keywords
        )

        if matched:
            categories.append(category)

    if not categories:
        categories.append("other")

    return categories


# ---------------------------------------------------------------------------
# RETEST / MOT EVENT GROUPING
# ---------------------------------------------------------------------------



def deduplicate_mot_tests(tests: list[dict]) -> list[dict]:
    """
    Consolidate duplicate MOT records that describe the same test moment.

    Historical bulk data can contain multiple records with the same
    completion timestamp, result and odometer reading. Rather than treating
    those as separate MOT events, merge their defects into one record.
    """

    merged: dict[tuple, dict] = {}

    for test in tests:
        completed_date = test.get("completedDate")
        result = test.get("testResult")
        mileage_miles = get_test_mileage_miles(test)

        key = (
            completed_date,
            result,
            mileage_miles,
        )

        if key not in merged:
            copied = dict(test)
            copied["defects"] = list(
                test.get("defects", [])
            )
            merged[key] = copied
            continue

        existing = merged[key]

        # Merge defect lists without duplicating the same description/type.
        existing_defects = existing.get("defects", [])
        seen_defects = {
            (
                normalise_defect_text(
                    defect.get("text", "")
                ),
                str(defect.get("type", "")).upper(),
                bool(defect.get("dangerous", False)),
            )
            for defect in existing_defects
        }

        for defect in test.get("defects", []):
            defect_key = (
                normalise_defect_text(
                    defect.get("text", "")
                ),
                str(defect.get("type", "")).upper(),
                bool(defect.get("dangerous", False)),
            )

            if defect_key not in seen_defects:
                existing_defects.append(defect)
                seen_defects.add(defect_key)

        existing["defects"] = existing_defects

        # Keep useful metadata when the first duplicate omitted it.
        for field in (
            "expiryDate",
            "registrationAtTimeOfTest",
            "regMarkTimeOfTest",
            "dataSource",
        ):
            if not existing.get(field) and test.get(field):
                existing[field] = test[field]

    return list(merged.values())


def is_probable_retest(event: dict, new_test: dict) -> bool:
    """
    Decide whether a new MOT test is probably a retest belonging to
    the previous failed MOT event.

    This is a historical-data heuristic, not an official confirmation
    that DVSA considers the test a retest.
    """

    # Only events that originally failed can collect a retest.
    if event.get("initial_result") != "FAILED":
        return False

    # Once an event finishes with a pass, consider it complete.
    if event.get("final_result") == "PASSED":
        return False

    if not event.get("tests"):
        return False

    previous_test = event["tests"][-1]

    previous_date = parse_datetime(
        previous_test.get("completedDate")
    )

    new_date = parse_datetime(
        new_test.get("completedDate")
    )

    if previous_date is None or new_date is None:
        return False

    day_difference = (
        new_date - previous_date
    ).total_seconds() / 86400

    if not 0 <= day_difference <= RETEST_WINDOW_DAYS:
        return False

    previous_mileage = get_test_mileage_miles(
        previous_test
    )

    new_mileage = get_test_mileage_miles(
        new_test
    )

    # If mileage is unavailable, use the date heuristic only.
    if previous_mileage is None or new_mileage is None:
        return True

    mileage_difference = abs(
        new_mileage - previous_mileage
    )

    return mileage_difference <= RETEST_MAX_MILES


def build_mot_events(vehicle_data: dict) -> list[dict]:
    """
    Convert individual DVSA MOT test records into MOT events.

    Example:

        FAILED test
             +
        PASSED retest a few days later
             =
        one MOT event

    Events are returned from oldest to newest.
    """

    mot_tests = vehicle_data.get("motTests", [])

    valid_tests = [
        test
        for test in mot_tests
        if parse_datetime(test.get("completedDate")) is not None
    ]

    valid_tests = deduplicate_mot_tests(
        valid_tests
    )

    tests = sorted(
        valid_tests,
        key=lambda test: parse_datetime(
            test["completedDate"]
        )
    )

    events = []

    for test in tests:

        if events and is_probable_retest(
            events[-1],
            test
        ):
            event = events[-1]

            event["tests"].append(test)

            event["final_result"] = test.get(
                "testResult"
            )

            event["end_date"] = test.get(
                "completedDate"
            )

            mileage = get_test_mileage_miles(
                test
            )

            if mileage is not None:
                event["end_mileage"] = mileage

            continue

        mileage = get_test_mileage_miles(
            test
        )

        event = {
            "initial_result": test.get(
                "testResult"
            ),

            "final_result": test.get(
                "testResult"
            ),

            "start_date": test.get(
                "completedDate"
            ),

            "end_date": test.get(
                "completedDate"
            ),

            "start_mileage": mileage,

            "end_mileage": mileage,

            "tests": [
                test
            ],
        }

        events.append(event)

    return events


# ---------------------------------------------------------------------------
# EVENT DEFECT HELPERS
# ---------------------------------------------------------------------------

def get_unique_event_defects(event: dict) -> list[dict]:
    """
    Return unique defects across every test in an MOT event.

    This is useful when reviewing the complete history of an event,
    but should NOT be used to represent the current/final condition
    after a successful retest.
    """

    unique = {}

    for test in event.get("tests", []):

        for defect in test.get("defects", []):

            text = defect.get("text", "")

            defect_type = defect.get(
                "type",
                ""
            ).upper()

            key = (
                normalise_defect_text(text),
                defect_type,
            )

            unique[key] = defect

    return list(unique.values())


def get_initial_test(event: dict) -> dict:
    """
    Return the first test in an MOT event.
    """

    tests = event.get("tests", [])

    if not tests:
        return {}

    return tests[0]


def get_final_test(event: dict) -> dict:
    """
    Return the final test/retest in an MOT event.
    """

    tests = event.get("tests", [])

    if not tests:
        return {}

    return tests[-1]


def get_initial_defects(event: dict) -> list[dict]:
    """
    Return defects recorded when the MOT event started.

    These are useful for historical-failure features because they show
    what caused the vehicle to fail before any repair/retest.
    """

    return get_initial_test(
        event
    ).get(
        "defects",
        []
    )


def get_final_defects(event: dict) -> list[dict]:
    """
    Return defects remaining when the MOT event finished.

    If a failed MOT was repaired and later passed, these final defects
    represent the condition recorded at the successful retest rather
    than the defects that caused the original failure.
    """

    return get_final_test(
        event
    ).get(
        "defects",
        []
    )


def analyse_defects(defects: list[dict]) -> dict:
    """
    Count defect severity and component categories for a list of defects.
    """

    advisory_count = 0
    major_count = 0
    dangerous_count = 0

    category_counts = Counter()

    for defect in defects:

        defect_type = defect.get(
            "type",
            ""
        ).upper()

        if defect_type == "ADVISORY":
            advisory_count += 1

        elif defect_type == "MAJOR":
            major_count += 1

        elif defect_type == "DANGEROUS":
            dangerous_count += 1

        text = defect.get(
            "text",
            ""
        )

        categories = categorise_defect(
            text
        )

        for category in categories:
            category_counts[category] += 1

    return {
        "advisories": advisory_count,
        "major": major_count,
        "dangerous": dangerous_count,
        "categories": dict(category_counts),
    }


def summarise_event(event: dict) -> dict:
    """
    Produce an API-friendly summary of one MOT event.

    The summary separates:
      - initial defects: what was present when the MOT event began
      - final defects: what remained when the event finished
    """

    initial_defects = get_initial_defects(
        event
    )

    final_defects = get_final_defects(
        event
    )

    initial_analysis = analyse_defects(
        initial_defects
    )

    final_analysis = analyse_defects(
        final_defects
    )

    return {
        "start_date": event.get(
            "start_date"
        ),

        "end_date": event.get(
            "end_date"
        ),

        "initial_result": event.get(
            "initial_result"
        ),

        "final_result": event.get(
            "final_result"
        ),

        "start_mileage": event.get(
            "start_mileage"
        ),

        "end_mileage": event.get(
            "end_mileage"
        ),

        "number_of_tests": len(
            event.get("tests", [])
        ),

        "was_retested": (
            len(event.get("tests", [])) > 1
        ),

        "initial_defects": {
            "advisories":
                initial_analysis[
                    "advisories"
                ],

            "major":
                initial_analysis[
                    "major"
                ],

            "dangerous":
                initial_analysis[
                    "dangerous"
                ],

            "categories":
                initial_analysis[
                    "categories"
                ],
        },

        "final_defects": {
            "advisories":
                final_analysis[
                    "advisories"
                ],

            "major":
                final_analysis[
                    "major"
                ],

            "dangerous":
                final_analysis[
                    "dangerous"
                ],

            "categories":
                final_analysis[
                    "categories"
                ],
        },
    }


# ---------------------------------------------------------------------------
# VEHICLE / MILEAGE FEATURES
# ---------------------------------------------------------------------------

def calculate_vehicle_age(
    first_used_date: Optional[str],
    as_of: Optional[datetime] = None,
) -> Optional[float]:
    """
    Calculate vehicle age in years.

    For live analysis, age is calculated at the current date.
    Later, when generating historical ML training rows, we will pass
    the prediction date as `as_of` to prevent future-data leakage.
    """

    if not first_used_date:
        return None

    try:
        first_used = datetime.fromisoformat(
            first_used_date
        )

    except (ValueError, TypeError):
        return None

    if as_of is None:
        as_of = datetime.now()

    age_days = (
        as_of.date()
        - first_used.date()
    ).days

    if age_days < 0:
        return None

    return round(
        age_days / 365.25,
        1
    )


def calculate_average_annual_mileage(
    events: list[dict],
) -> Optional[int]:
    """
    Estimate average annual mileage using the first and latest
    usable MOT-event mileage readings.
    """

    usable = [
        event
        for event in events
        if event.get("end_mileage") is not None
        and parse_datetime(
            event.get("end_date")
        ) is not None
    ]

    if len(usable) < 2:
        return None

    first = usable[0]
    last = usable[-1]

    first_date = parse_datetime(
        first.get("end_date")
    )

    last_date = parse_datetime(
        last.get("end_date")
    )

    if first_date is None or last_date is None:
        return None

    days = (
        last_date - first_date
    ).days

    mileage_difference = (
        last["end_mileage"]
        - first["end_mileage"]
    )

    if days <= 0:
        return None

    if mileage_difference < 0:
        return None

    years = days / 365.25

    if years <= 0:
        return None

    return round(
        mileage_difference / years
    )


# ---------------------------------------------------------------------------
# ADVISORY / HISTORY HELPERS
# ---------------------------------------------------------------------------

def event_has_advisory_category(
    event: dict,
    category_name: str,
) -> bool:
    """
    Return True when an advisory in the requested category remains
    on the FINAL test of the MOT event.

    This avoids treating repaired failure defects as current advisories.
    """

    defects = get_final_defects(
        event
    )

    for defect in defects:

        if defect.get(
            "type",
            ""
        ).upper() != "ADVISORY":
            continue

        categories = categorise_defect(
            defect.get(
                "text",
                ""
            )
        )

        if category_name in categories:
            return True

    return False


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def extract_mot_features(
    vehicle_data: dict,
) -> dict:
    """
    Convert complete MOT history into machine-learning-friendly features.

    Important distinction:

    CURRENT / RECENT CONDITION FEATURES
        These use final-test defects. If a failed MOT was repaired and
        passed on retest, repaired major/dangerous defects are not treated
        as still being present.

    HISTORICAL FAILURE FEATURES
        These use initial-test defects so the model can still learn that
        the vehicle recently experienced serious failures.
    """

    events = build_mot_events(
        vehicle_data
    )

    if not events:
        return {
            "mot_event_count": 0
        }

    summaries = [
        summarise_event(event)
        for event in events
    ]

    total_initial_failures = sum(
        1
        for event in events
        if event.get(
            "initial_result"
        ) == "FAILED"
    )

    total_initial_passes = sum(
        1
        for event in events
        if event.get(
            "initial_result"
        ) == "PASSED"
    )

    retest_events = sum(
        1
        for event in events
        if len(
            event.get(
                "tests",
                []
            )
        ) > 1
    )

    latest = events[-1]

    previous = (
        events[-2]
        if len(events) >= 2
        else None
    )

    latest_mileage = latest.get(
        "end_mileage"
    )

    previous_mileage = (
        previous.get(
            "end_mileage"
        )
        if previous
        else None
    )

    mileage_since_previous_mot = None

    if (
        latest_mileage is not None
        and previous_mileage is not None
    ):
        difference = (
            latest_mileage
            - previous_mileage
        )

        if difference >= 0:
            mileage_since_previous_mot = (
                difference
            )

    features = {
        "vehicle_age_years":
            calculate_vehicle_age(
                vehicle_data.get(
                    "firstUsedDate"
                )
            ),

        "latest_mileage":
            latest_mileage,

        "average_annual_mileage":
            calculate_average_annual_mileage(
                events
            ),

        "mileage_since_previous_mot":
            mileage_since_previous_mot,

        "mot_event_count":
            len(events),

        "initial_pass_count":
            total_initial_passes,

        "initial_fail_count":
            total_initial_failures,

        "historical_failure_rate":
            round(
                total_initial_failures
                / len(events),
                3
            ),

        "retest_event_count":
            retest_events,

        "latest_initial_test_failed":
            int(
                latest.get(
                    "initial_result"
                ) == "FAILED"
            ),

        "latest_final_result_passed":
            int(
                latest.get(
                    "final_result"
                ) == "PASSED"
            ),
    }

    # -----------------------------------------------------------------------
    # RECENT MOT OUTCOME HISTORY
    # -----------------------------------------------------------------------

    last_three = events[-3:]

    features[
        "failures_last_3_events"
    ] = sum(
        1
        for event in last_three
        if event.get(
            "initial_result"
        ) == "FAILED"
    )

    # -----------------------------------------------------------------------
    # HISTORICAL SERIOUS-FAILURE FEATURES
    #
    # These intentionally use INITIAL defects so repaired failures remain
    # useful as evidence of vehicle history.
    # -----------------------------------------------------------------------

    historical_major_last_3 = 0
    historical_dangerous_last_3 = 0

    historical_failure_categories = Counter()

    for event in last_three:

        initial_defects = get_initial_defects(
            event
        )

        for defect in initial_defects:

            defect_type = defect.get(
                "type",
                ""
            ).upper()

            if defect_type == "MAJOR":
                historical_major_last_3 += 1

            elif defect_type == "DANGEROUS":
                historical_dangerous_last_3 += 1

            # Only major/dangerous defects count as historical
            # failure-category features.
            if defect_type not in {
                "MAJOR",
                "DANGEROUS",
            }:
                continue

            categories = categorise_defect(
                defect.get(
                    "text",
                    ""
                )
            )

            for category in categories:
                historical_failure_categories[
                    category
                ] += 1

    features[
        "historical_major_defects_last_3"
    ] = historical_major_last_3

    features[
        "historical_dangerous_defects_last_3"
    ] = historical_dangerous_last_3

    all_categories = (
        list(
            DEFECT_CATEGORIES.keys()
        )
        + ["other"]
    )

    for category in all_categories:

        features[
            f"historical_{category}_failures_last_3"
        ] = historical_failure_categories.get(
            category,
            0,
        )

    # -----------------------------------------------------------------------
    # LATEST FINAL MOT STATE
    #
    # These use FINAL defects because a successful retest may have repaired
    # the major/dangerous defects from the initial failed test.
    # -----------------------------------------------------------------------

    latest_summary = summaries[-1]

    features[
        "latest_advisory_count"
    ] = latest_summary[
        "final_defects"
    ][
        "advisories"
    ]

    features[
        "latest_major_count"
    ] = latest_summary[
        "final_defects"
    ][
        "major"
    ]

    features[
        "latest_dangerous_count"
    ] = latest_summary[
        "final_defects"
    ][
        "dangerous"
    ]

    # -----------------------------------------------------------------------
    # RECENT COMPONENT-CONDITION FEATURES
    #
    # Use only the FINAL defects of the latest two MOT events. This avoids
    # treating repaired major/dangerous failures as current defects.
    # -----------------------------------------------------------------------

    recent_events = events[-2:]

    for category in all_categories:

        recent_defect_count = 0
        recent_advisory_count = 0
        recent_major_count = 0
        recent_dangerous_count = 0

        for event in recent_events:

            defects = get_final_defects(
                event
            )

            for defect in defects:

                defect_categories = (
                    categorise_defect(
                        defect.get(
                            "text",
                            ""
                        )
                    )
                )

                if category not in defect_categories:
                    continue

                recent_defect_count += 1

                defect_type = defect.get(
                    "type",
                    ""
                ).upper()

                if defect_type == "ADVISORY":
                    recent_advisory_count += 1

                elif defect_type == "MAJOR":
                    recent_major_count += 1

                elif defect_type == "DANGEROUS":
                    recent_dangerous_count += 1

        features[
            f"{category}_defects_last_2"
        ] = recent_defect_count

        features[
            f"{category}_advisories_last_2"
        ] = recent_advisory_count

        features[
            f"{category}_major_last_2"
        ] = recent_major_count

        features[
            f"{category}_dangerous_last_2"
        ] = recent_dangerous_count

        # Did the same advisory category remain on the final test
        # of BOTH of the latest two MOT events?
        repeated = 0

        if len(events) >= 2:

            repeated = int(
                event_has_advisory_category(
                    events[-1],
                    category,
                )
                and
                event_has_advisory_category(
                    events[-2],
                    category,
                )
            )

        features[
            f"repeated_{category}_advisory"
        ] = repeated

    return features