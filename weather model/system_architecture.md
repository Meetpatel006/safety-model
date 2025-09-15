## Weather Safety Model — System Architecture

### Overview

This document describes the architecture for the weather-based safety model component. It covers data sources, feature engineering, model training, serving, deployment, and monitoring for the code in the `weather model/` folder.

### High-level components

- Data sources
  - `weatherHistory.csv` — historical weather observations used for training
  - External APIs (optional): NOAA, OpenWeatherMap for live data
  - `feature_importance.csv` — recorded importance from experiments
- Data processing
  - Feature extraction scripts (in `feature_importance.csv` and `training.py`) that prepare lag features, rolling averages, and categorical encodings
- Model training
  - `training.py` — training pipeline that outputs `weather_safety_model_kaggle.pkl`
  - Training logs: `weather_model_training.log`
- Model serving / API
  - `api.py` — exposes endpoints for predictions and model metadata
  - `use_api.py` — helper script showing client-side usage
- Persistence
  - Model artifacts serialized as pickle files and CSVs for feature importance

### Dataflow

1. Raw weather data pulled from `weatherHistory.csv` or external APIs.
2. `training.py` builds features (lags, rolling means, categorical encodings), trains a model, and serializes the model and optionally a tokenizer/encoder.
3. `api.py` loads model artifact(s) and serves prediction endpoints for online inference.

Simple ASCII flow:

weatherHistory.csv / API -> training.py -> weather_safety_model_kaggle.pkl
                                         |
                                         v
                                      api.py (loads model)

### Key API endpoints (suggested)

- POST /predict
  - Input: JSON with current weather summary or time series window
  - Output: safety/risk score, top features
- GET /model
  - Output: model metadata and evaluation metrics

Refer to `api.py` for exact routes and payload shapes.

### Data formats

- Input time series: JSON array of timestamped observations or a single aggregated snapshot
- Training CSV: rows with timestamped measurements and labels
- Model artifacts: pickled sklearn or xgboost objects

### Feature engineering notes

- Time-based features: hour-of-day, day-of-week, seasonal flags
- Lag features and rolling statistics: 1h/3h/24h averages, differences
- Categorical encoding for weather conditions (sunny, rainy, snow)
- Normalization: consistent scaling between training and serving using saved scalers

### Model and explainability

- Model type: tree-based models (XGBoost/LightGBM) or scikit-learn classifiers/ regressors.
- Store feature importance CSVs and provide per-prediction explanations when possible.

### Deployment

- Containerize with a Dockerfile based on Python image and `requirements.txt`.
- Serve via a WSGI server (uvicorn/gunicorn) depending on framework (FastAPI/Flask).
- Keep model artifacts in an immutable storage (S3, GCS) and download on container start.

#### Serverless (Modal)

- Modal can be used to deploy serverless Python functions for the weather model. Modal provides an HTTP gateway and a secrets store so you can run inference without managing servers.
- Store model artifacts in object storage and have Modal functions fetch and cache them at cold-start. Use Modal's filesystem layers or a small initialization function to download models once per container instance.
- Cold-start and size: for predictable latency, keep model artifacts compact or use lazy-loading strategies. For larger models, pre-warm instances or host the heavy model in a separate service.
- Routing & timeouts: configure Modal HTTP endpoints to match expected request/response patterns and set safe timeouts for long-running batch jobs (offload to workers if necessary).