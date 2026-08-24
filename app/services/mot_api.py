import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()


CLIENT_ID = os.getenv("MOT_CLIENT_ID")
CLIENT_SECRET = os.getenv("MOT_CLIENT_SECRET")
API_KEY = os.getenv("MOT_API_KEY")

SCOPE = os.getenv("MOT_SCOPE")
TOKEN_URL = os.getenv("MOT_TOKEN_URL")
BASE_URL = os.getenv(
    "MOT_API_BASE_URL",
    "https://history.mot.api.gov.uk"
)


_access_token = None
_token_expiry = 0


async def get_access_token():
    global _access_token
    global _token_expiry

    # Return cached token if it is still valid
    if _access_token and time.time() < _token_expiry:
        return _access_token

    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": SCOPE,
    }

    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            TOKEN_URL,
            data=data,
            headers=headers,
        )

    response.raise_for_status()

    token_data = response.json()

    _access_token = token_data["access_token"]

    expires_in = token_data.get("expires_in", 3600)

    # Refresh slightly before actual expiry
    _token_expiry = time.time() + expires_in - 60

    return _access_token


async def get_vehicle_mot_history(registration: str):

    token = await get_access_token()

    registration = registration.replace(" ", "").upper()

    url = (
        f"{BASE_URL}"
        f"/v1/trade/vehicles/registration/{registration}"
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "X-API-Key": API_KEY,
        "Accept": "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            headers=headers,
        )

    response.raise_for_status()

    return response.json()