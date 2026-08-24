MOT Health Predictor Frontend

Copy:
  index.html
  styles.css
  app.js

into:
  app/static/

Then add these imports to app/main.py:

from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

After the FastAPI app object is created, add:

STATIC_DIR = Path(__file__).parent / "static"

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

Then add:

@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    return FileResponse(STATIC_DIR / "index.html")

Restart:
  uvicorn app.main:app --reload

Open:
  http://127.0.0.1:8000/dashboard

The page calls:
  GET /vehicle/{registration}/report

Because both frontend and API are served by FastAPI on the same origin,
no CORS configuration is required for local development.
