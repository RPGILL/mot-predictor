import re


def normalise_registration(registration: str) -> str:

    registration = registration.upper()

    registration = re.sub(
        r"[^A-Z0-9]",
        "",
        registration
    )

    return registration