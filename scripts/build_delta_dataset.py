from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Iterator, Optional

import httpx
from dotenv import load_dotenv

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.mot_api import get_access_token
from app.services.training_data import build_training_rows_for_vehicle

load_dotenv()

BASE_URL = os.getenv("MOT_API_BASE_URL", "https://history.mot.api.gov.uk")
API_KEY = os.getenv("MOT_API_KEY")
MANIFEST_ENDPOINT = f"{BASE_URL}/v1/trade/vehicles/bulk-download"


def require_api_key() -> str:
    if not API_KEY:
        raise RuntimeError("MOT_API_KEY is missing from .env")
    return API_KEY


async def get_download_manifest() -> dict:
    token = await get_access_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "X-API-Key": require_api_key(),
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(
            MANIFEST_ENDPOINT,
            headers=headers,
        )

    response.raise_for_status()
    return response.json()


def choose_latest_light_vehicle_delta(manifest: dict) -> dict:
    deltas = manifest.get("delta", [])

    if not deltas:
        raise RuntimeError("DVSA returned no delta files.")

    light_vehicle_deltas = [
        item
        for item in deltas
        if "light-vehicle" in item.get("filename", "").lower()
    ]

    candidates = light_vehicle_deltas or deltas

    return max(
        candidates,
        key=lambda item: item.get("fileCreatedOn", ""),
    )


async def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=None,
    ) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()

            with destination.open("wb") as output:
                async for chunk in response.aiter_bytes(
                    chunk_size=1024 * 1024
                ):
                    output.write(chunk)

    return destination


def _iter_json_lines(binary_stream) -> Iterator[dict]:
    text_stream = io.TextIOWrapper(
        binary_stream,
        encoding="utf-8",
        errors="replace",
    )

    for line in text_stream:
        line = line.strip()

        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        if isinstance(record, dict):
            yield record


def iter_vehicle_records(archive_path: Path) -> Iterator[dict]:
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            member_name = member.filename.lower()

            with archive.open(member, "r") as raw_member:
                if member_name.endswith(".gz"):
                    with gzip.GzipFile(
                        fileobj=raw_member,
                        mode="rb",
                    ) as gzip_stream:
                        yield from _iter_json_lines(gzip_stream)

                elif (
                    member_name.endswith(".json")
                    or member_name.endswith(".jsonl")
                    or member_name.endswith(".ndjson")
                ):
                    yield from _iter_json_lines(raw_member)


def has_usable_dvsa_history(
    vehicle: dict,
    minimum_tests: int = 2,
) -> bool:
    modification = (
        vehicle.get("modification", "")
        or vehicle.get("last_modification", "")
    )

    if str(modification).upper() == "DELETED":
        return False

    mot_tests = vehicle.get("motTests", [])

    if not isinstance(mot_tests, list):
        return False

    dvsa_tests = [
        test
        for test in mot_tests
        if str(test.get("dataSource", "")).lower() == "dvsa"
        and test.get("testResult") in {"PASSED", "FAILED"}
    ]

    return len(dvsa_tests) >= minimum_tests


def keep_dvsa_tests_only(vehicle: dict) -> dict:
    cleaned = dict(vehicle)

    cleaned["motTests"] = [
        test
        for test in vehicle.get("motTests", [])
        if str(test.get("dataSource", "")).lower() == "dvsa"
        and test.get("testResult") in {"PASSED", "FAILED"}
    ]

    return cleaned


def build_csv_from_archive(
    archive_path: Path,
    output_csv: Path,
    max_vehicles: Optional[int] = 5000,
    min_history_events: int = 2,
) -> dict:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    vehicles_seen = 0
    vehicles_used = 0
    training_rows_written = 0

    writer = None

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_handle:

        for vehicle in iter_vehicle_records(archive_path):
            vehicles_seen += 1

            if not has_usable_dvsa_history(vehicle):
                continue

            vehicle = keep_dvsa_tests_only(vehicle)

            rows = build_training_rows_for_vehicle(
                vehicle,
                min_history_events=min_history_events,
            )

            if not rows:
                continue

            vehicles_used += 1

            for row in rows:
                if writer is None:
                    writer = csv.DictWriter(
                        output_handle,
                        fieldnames=list(row.keys()),
                    )
                    writer.writeheader()

                writer.writerow(row)
                training_rows_written += 1

            if vehicles_used % 100 == 0:
                output_handle.flush()

                print(
                    f"Processed {vehicles_used:,} usable vehicles; "
                    f"{training_rows_written:,} training rows."
                )

            if (
                max_vehicles is not None
                and vehicles_used >= max_vehicles
            ):
                break

    return {
        "vehicles_seen": vehicles_seen,
        "vehicles_used": vehicles_used,
        "training_rows_written": training_rows_written,
        "output_csv": str(output_csv),
    }


async def download_latest_delta(output_directory: Path) -> Path:
    manifest = await get_download_manifest()
    delta = choose_latest_light_vehicle_delta(manifest)

    filename = Path(
        delta.get("filename", "latest_delta.zip")
    ).name

    download_url = delta.get("downloadUrl")

    if not download_url:
        raise RuntimeError(
            "The selected DVSA delta does not contain a downloadUrl."
        )

    destination = output_directory / filename

    print("Downloading DVSA delta:", filename)

    if delta.get("fileSize") is not None:
        print(
            "Reported compressed size:",
            f"{int(delta['fileSize']):,}",
            "bytes",
        )

    await download_file(download_url, destination)

    print("Downloaded to:", destination)

    return destination


async def main_async(
    max_vehicles: Optional[int],
    min_history_events: int,
    archive_path: Optional[str],
    output_csv: str,
) -> None:
    raw_directory = Path("data/raw")

    if archive_path:
        archive = Path(archive_path)

        if not archive.exists():
            raise FileNotFoundError(
                f"Archive not found: {archive}"
            )
    else:
        archive = await download_latest_delta(raw_directory)

    result = build_csv_from_archive(
        archive_path=archive,
        output_csv=Path(output_csv),
        max_vehicles=max_vehicles,
        min_history_events=min_history_events,
    )

    print()
    print("Dataset build complete.")
    print("Vehicle records scanned:", f"{result['vehicles_seen']:,}")
    print("Usable vehicles:", f"{result['vehicles_used']:,}")
    print("Training rows:", f"{result['training_rows_written']:,}")
    print("CSV:", result["output_csv"])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download a DVSA daily delta and build "
            "a leakage-free MOT training CSV."
        )
    )

    parser.add_argument(
        "--max-vehicles",
        type=int,
        default=5000,
        help=(
            "Maximum usable vehicles to process. "
            "Use 0 for no limit. Default: 5000."
        ),
    )

    parser.add_argument(
        "--min-history-events",
        type=int,
        default=2,
        help=(
            "Historical MOT events required before creating a target row. "
            "Default: 2."
        ),
    )

    parser.add_argument(
        "--archive",
        type=str,
        default=None,
        help=(
            "Optional already-downloaded DVSA delta ZIP. "
            "If omitted, the newest light-vehicle delta is downloaded."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/mot_training_sample.csv",
        help=(
            "Output CSV path. "
            "Default: data/processed/mot_training_sample.csv"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    max_vehicles = (
        None
        if args.max_vehicles == 0
        else args.max_vehicles
    )

    asyncio.run(
        main_async(
            max_vehicles=max_vehicles,
            min_history_events=args.min_history_events,
            archive_path=args.archive,
            output_csv=args.output,
        )
    )


if __name__ == "__main__":
    main()