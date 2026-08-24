# MOT Health Predictor

MOT Health Predictor is a machine-learning project that estimates whether a vehicle may pass or fail its next initial MOT test.

The user enters a GB vehicle registration. The application retrieves the vehicle's MOT history, creates prediction features, runs two trained models and displays an explainable report.

> This is an academic project. It does not replace a real vehicle inspection and cannot guarantee an MOT result.

## Main features

- Retrieves vehicle MOT history.
- Predicts the probability of an MOT pass or failure.
- Predicts possible failure areas.
- Uses previous defects and advisories to explain the result.
- Displays likely failure areas.
- Displays current MOT advisories.
- Provides recommended vehicle checks.
- Includes a FastAPI backend.
- Includes an HTML, CSS and JavaScript dashboard.

## How it works

1. The user enters a vehicle registration.
2. The application retrieves the vehicle's MOT history.
3. The history is changed into model features.
4. The PASS/FAIL model calculates the failure probability.
5. The category model predicts possible problem areas.
6. The explanation service checks the predictions against the real MOT history.
7. The dashboard displays the final report.

## Machine-learning models

The application uses two saved models.

| Model | Algorithm | Purpose |
|---|---|---|
| `mot_pass_fail_model.joblib` | Random Forest | Predicts the overall PASS and FAIL probabilities |
| `mot_failure_category_model.joblib` | One-vs-Rest Logistic Regression | Predicts possible MOT failure categories |

The failure-category model covers:

- Tyres
- Brakes
- Suspension
- Lighting
- Steering
- Visibility
- Emissions and exhaust
- Body and structure
- Seatbelts and SRS
- Other defects

The PASS/FAIL training process compared:

- Dummy Classifier
- Logistic Regression
- Random Forest

Random Forest was selected because it had the best validation ROC-AUC result.

## Technology used

- Python
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Joblib
- HTTPX
- HTML
- CSS
- JavaScript

## Project structure

```text
mot-predictor/
├── app/
│   ├── main.py
│   ├── services/
│   │   ├── mot_api.py
│   │   ├── feature_engineering.py
│   │   ├── predictor.py
│   │   └── explainable_report.py
│   └── static/
│       ├── index.html
│       ├── styles.css
│       └── app.js
├── data/
├── models/
├── reports/
├── scripts/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/mot-health-predictor.git
cd mot-health-predictor
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install the packages

```powershell
pip install -r requirements.txt
```

## Environment variables

Create a file named `.env` in the main project folder.

```env
MOT_CLIENT_ID=your_client_id
MOT_CLIENT_SECRET=your_client_secret
MOT_API_KEY=your_api_key
MOT_SCOPE=your_scope
MOT_TOKEN_URL=your_token_url
MOT_API_BASE_URL=https://history.mot.api.gov.uk
```

Never upload the `.env` file to GitHub.

## Model files

The trained model files are excluded from GitHub because the main model is larger than GitHub's normal file-size limit.

Before running predictions, place these files inside the `models` folder:

```text
models/mot_pass_fail_model.joblib
models/mot_failure_category_model.joblib
```

The model feature JSON files must also remain inside the `models` folder.

The threshold JSON files must remain inside the `reports` folder.

## Run the application

Run this command from the main project folder:

```powershell
uvicorn app.main:app --reload
```

Open the dashboard:

```text
http://127.0.0.1:8000/dashboard
```

Open the FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## Main API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Checks whether the API is running |
| `GET` | `/vehicle/{registration}` | Retrieves the MOT history |
| `GET` | `/vehicle/{registration}/features` | Shows the model features |
| `GET` | `/vehicle/{registration}/analysis` | Shows the vehicle analysis |
| `GET` | `/vehicle/{registration}/prediction` | Returns the PASS/FAIL prediction |
| `GET` | `/vehicle/{registration}/report` | Returns the explainable MOT report |
| `GET` | `/dashboard` | Opens the web dashboard |

## Train the models

Training requires the prepared dataset:

```text
data/processed/mot_training_sample.csv
```

Run these commands from the main project folder:

```powershell
python scripts/train_model.py
python scripts/tune_threshold.py
python scripts/train_category_model.py
python scripts/tune_category_thresholds.py
```

The trained models are saved inside the `models` folder.

The evaluation results are saved inside the `reports` folder.

## Important limitations

- The application uses past MOT history.
- It cannot physically inspect the vehicle.
- New faults may not appear in the MOT history.
- A high risk does not guarantee failure.
- A low risk does not guarantee a pass.
- Category predictions may be uncertain.
- The result should only be used as guidance for a pre-MOT check.
