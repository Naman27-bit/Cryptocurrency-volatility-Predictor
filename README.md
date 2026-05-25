# Crypto Volatility Predictor (Streamlit + FastAPI)

This project exposes an ML model (trained from `dataset.csv`) to predict **`volatility_14`** using a small set of engineered features.

- **FastAPI** provides a REST endpoint: `POST /predict`
- **Streamlit** provides a simple UI to enter the 5 engineered features and show the prediction.

> Notes
> - The current service expects **engineered inputs**: `volatility_7`, `ma_7`, `ma_14`, `hl_range`, `volume_mcap_ratio`.
> - On first request, the model is trained and cached in-memory (process-local) via `functools.lru_cache`.

---

## Project structure

- `api_fastapi.py` – FastAPI app
- `streamlit_app.py` – Streamlit UI
- `app_service/model_service.py` – feature engineering + model training + prediction helpers
- `dataset.csv` – training dataset
- `requirements.txt` – Python dependencies

---

## Setup

1) Create/activate a virtual environment (if you don’t already have one):

```bash
python -m venv venv
.
venv\Scripts\Activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run FastAPI

```bash
uvicorn api_fastapi:app --reload --port 8000
```

Health check:

- `GET http://127.0.0.1:8000/health`

Prediction:

- `POST http://127.0.0.1:8000/predict`

Example request body:

```json
{
  "volatility_7": 0.01,
  "ma_7": 100.0,
  "ma_14": 98.0,
  "hl_range": 0.02,
  "volume_mcap_ratio": 0.5
}
```

Optional:

- `dataset_path` (defaults to `dataset.csv`)

---

## Run Streamlit

```bash
streamlit run streamlit_app.py --server.port 8501
```

Open:

- `http://localhost:8501`

---

## Dependencies

Key libraries:

- `fastapi`, `uvicorn`, `pydantic`
- `streamlit`
- `pandas`, `numpy`
- `xgboost`, `scikit-learn`

---

## Troubleshooting

- If FastAPI import fails, ensure you installed dependencies inside the active venv.
- First prediction may take longer (model training happens on-demand).


