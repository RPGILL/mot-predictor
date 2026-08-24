import httpx
from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from app.services.training_data import (
    build_training_rows_for_vehicle,
)
from app.services.predictor import (
    predict_mot_risk,
)
from app.services.explainable_report import (
    build_explainable_mot_report,
)
from app.services.mot_api import get_vehicle_mot_history
from app.services.report_service import normalise_registration
from app.services.feature_engineering import extract_mot_features
from app.services.feature_engineering import (
    build_mot_events,
    extract_mot_features,
    summarise_event,
)

from app.services.health_report import (
    build_health_report,
)

app = FastAPI(
    title="MOT Health Predictor",
    description="MOT readiness and vehicle health prediction system",
    version="0.1.0",
)
STATIC_DIR = (
    Path(__file__).parent
    / "static"
)

app.mount(
    "/static",
    StaticFiles(
        directory=STATIC_DIR
    ),
    name="static",
)

@app.get(
    "/dashboard",
    include_in_schema=False,
)
async def dashboard():

    return FileResponse(
        STATIC_DIR
        / "index.html"
    )

@app.get("/")
def home():
    return {
        "name": "MOT Health Predictor",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/vehicle/{registration}")
async def get_vehicle(registration: str):

    registration = normalise_registration(registration)

    try:
        vehicle = await get_vehicle_mot_history(registration)
        return vehicle

    except httpx.HTTPStatusError as error:
        status = error.response.status_code

        if status == 404:
            raise HTTPException(
                status_code=404,
                detail="Vehicle not found"
            )

        if status in [401, 403]:
            raise HTTPException(
                status_code=status,
                detail="DVSA authentication failed"
            )

        if status == 429:
            raise HTTPException(
                status_code=429,
                detail="DVSA API rate limit reached"
            )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve MOT history"
        )


@app.get("/vehicle/{registration}/features")
async def get_vehicle_features(registration: str):

    registration = normalise_registration(registration)

    try:
        vehicle_data = await get_vehicle_mot_history(registration)
        features = extract_mot_features(vehicle_data)

        return {
            "registration": registration,
            "make": vehicle_data.get("make"),
            "model": vehicle_data.get("model"),
            "features": features
        }

    except httpx.HTTPStatusError:
        raise HTTPException(
            status_code=500,
            detail="Unable to calculate vehicle features"
        )
@app.get("/vehicle/{registration}/analysis")
async def analyse_vehicle(
    registration: str
):

    registration = normalise_registration(
        registration
    )

    try:

        vehicle_data = (
            await get_vehicle_mot_history(
                registration
            )
        )

        events = build_mot_events(
            vehicle_data
        )

        event_summaries = [
            summarise_event(event)
            for event in events
        ]

        features = extract_mot_features(
            vehicle_data
        )

        health_report = build_health_report(
            features
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

            "mot_events":
                event_summaries,

            "features":
                features,

            "health_report":
                health_report,
        }

    except httpx.HTTPStatusError as error:

        status = error.response.status_code

        if status == 404:
            raise HTTPException(
                status_code=404,
                detail="Vehicle not found",
            )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to analyse vehicle"
            ),
        )
@app.get(
    "/vehicle/{registration}/training-rows"
)
async def get_training_rows(
    registration: str
):

    registration = normalise_registration(
        registration
    )

    try:

        vehicle_data = (
            await get_vehicle_mot_history(
                registration
            )
        )

        rows = (
            build_training_rows_for_vehicle(
                vehicle_data
            )
        )

        return {
            "registration": registration,
            "training_row_count": len(rows),
            "training_rows": rows,
        }

    except httpx.HTTPStatusError as error:

        status = (
            error.response.status_code
        )

        if status == 404:

            raise HTTPException(
                status_code=404,
                detail="Vehicle not found",
            )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate "
                "training rows"
            ),
        )
@app.get(
    "/vehicle/{registration}/prediction"
)
async def predict_vehicle_mot(
    registration: str
):

    registration = normalise_registration(
        registration
    )

    try:

        vehicle_data = (
            await get_vehicle_mot_history(
                registration
            )
        )

        prediction = predict_mot_risk(
            vehicle_data
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

            "mot_prediction":
                prediction,
        }

    except httpx.HTTPStatusError as error:

        status = (
            error.response.status_code
        )

        if status == 404:
            raise HTTPException(
                status_code=404,
                detail="Vehicle not found",
            )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate "
                "MOT prediction"
            ),
        )

    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
@app.get(
    "/vehicle/{registration}/report"
)
async def get_vehicle_report(
    registration: str
):

    registration = normalise_registration(
        registration
    )

    try:

        vehicle_data = (
            await get_vehicle_mot_history(
                registration
            )
        )

        return build_explainable_mot_report(
            vehicle_data
        )

    except httpx.HTTPStatusError as error:

        status = (
            error.response.status_code
        )

        if status == 404:

            raise HTTPException(
                status_code=404,
                detail="Vehicle not found",
            )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate "
                "MOT health report"
            ),
        )

    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )