## Geo-fencing And Weather Models

### Purpose

This document unifies the system flow for both the Geo-fencing and Weather safety models in this repository and describes integration patterns and recommended deployment approaches. Important: the models in this repository are deployed independently (separate services/endpoints) — they are not packaged or deployed as a single combined model. The doc still shows a conceptual combined flow to explain how scores could be aggregated, but any aggregator is optional and not implemented in this repo.

### High-level components (shared)

- Data stores: object storage (S3/GCS) for model artifacts and CSV/JSON datasets.
- Secrets store: Modal secrets or environment variables for external API keys.
- Inference endpoints: one or more HTTP endpoints for the geo-fencing and weather models.
- Orchestration: optional queue/worker system for batch scoring (Celery/RQ) or Modal scheduled functions.

### Per-component summary

- Geo-fencing
  - Inputs: `geofences.json`, route/points, geofence-specific features CSVs
  - Output: geofence risk score, geofence id, contributing features
  - Artifacts: `geofence_safety_model.pkl`, `label_encoder.pkl`

- Weather
  - Inputs: `weatherHistory.csv`, live weather API data
  - Output: weather-based risk score, top weather features
  - Artifacts: `weather_safety_model_kaggle.pkl`, `feature_importance.csv`

### Combined dataflow

1. Data ingestion
   - Geofence definitions and historical CSVs are stored in the repo or object storage.
   - Weather data is stored or pulled from external APIs and normalized.
2. Feature engineering
   - Geo-fencing: spatial and temporal features (distance to boundary, dwell time, speed)
   - Weather: time-series features (lags, rolling averages, categorical encodings)
3. Training
   - Each pipeline produces a serialized model artifact and feature metadata.
   - Store artifacts in object storage and tag versions with timestamps/commit hashes.
4. Serving
   - Deploy inference endpoints for each model. In this repository the Geo-fencing and Weather models are served as separate endpoints (separate services/functions). They are not deployed as a single combined model.
   - Clients may call each model independently (for example: one call for geo-fencing, one for weather). If a combined score is desired, an external aggregator or gateway can call both endpoints and merge the results — this aggregator is optional and not part of the current repo.

Project-accurate ASCII flow (reflects repository layout; aggregator optional/out-of-repo):

            +-----------------+                      +--------------------+
            |  Geo data &     |                      |  Weather data &    |
            |  geofences.json |                      |  weatherHistory.csv|
            +--------+--------+                      +----------+---------+
                     |                                      |
                     v                                      v
            +-----------------+                      +--------------------+
            | geo fencing/    |                      | weather model/     |
            | (feature.py,    |                      | (feature_importance,
            |  dataset.py,    |                      |  training.py, api.py)
            |  training.py)   |                      |                    |
            +--------+--------+                      +----------+---------+
                     |                                      |
                     v                                      v
            +-----------------+                      +--------------------+
            | geofence_safety |                      | weather_safety_model|
            | _model.pkl,     |                      | _kaggle.pkl         |
            | label_encoder.pkl|                     | feature_importance.csv
            +--------+--------+                      +----------+---------+
                     |                                      |
                     v                                      v
            +-----------------+                      +--------------------+
            | Geo-fencing API |                      | Weather API        |
            | (e.g. /predict) |                      | (e.g. /predict)    |
            +--------+--------+                      +----------+---------+
                     \                                      /
                      \                                    /
                       -> (Optional) Aggregator / Client <-

Note: The repository contains the separate `geo fencing/` and `weather model/` folders, their model artifacts, and example serving code. A combined aggregator (if used) should be implemented as a separate service and is not included here.

### API design patterns

- Keep endpoints small and focused. Note: both models use the same endpoint patterns (for example `POST /predict`, `GET /model`) but they are exposed under different base URLs so the URLs differ while the payload/response contracts remain consistent:
   - `POST /predict` — route or point sequence -> geofence risk
   - `GET /model` — geo model metadata
   - `POST /predict` — weather snapshot or time-series -> weather risk
   - `GET /model` — weather model metadata
- Responses should include: score, model version, feature contributions (top-n), and timestamp.

### Deployment on Modal (serverless)

Modal is well suited for hosting light-weight API endpoints and scheduled/batch jobs. Below are best practices and a recommended architecture — note that the examples in this repository assume separate deployments for each model (separate functions/endpoints), not a single combined service.

1. Model storage & versioning
   - Keep model artifacts in an object store (S3/GCS) or Modal's file layers. Tag artifacts with semantic versions or commit hashes.
   - At cold-start, a Modal function should check artifact version and download only when missing or out-of-date. Cache the artifact in the filesystem for the lifetime of the container instance.

2. Secrets & credentials
   - Use Modal's secrets store to keep external API keys and storage credentials. Inject them as environment variables into the function runtime.

3. Function layout
   - In this repo, use separate Modal functions for `geo_predict` and `weather_predict` to keep functions small and specialized. This mirrors the repo layout (separate model artifacts and serving code).
   - If you need a combined risk score, implement an external `aggregator` service or gateway that calls both endpoints — the aggregator is optional and intentionally out-of-repo.

4. HTTP routing & gateway
   - Use Modal's HTTP gateway to expose endpoints. Expose `/predict` under each model's base path (for example `/predict` and `/predict`).

5. Cold-start & latency mitigation
   - Keep model artifacts compact (<50MB suggested) or use lazy loading for model components.
   - For predictable latency, maintain a warm pool of Modal instances or use Modal's scheduling to periodically invoke functions.

6. Observability
   - Emit logs, metrics, and model version as part of each prediction. Sample and store inputs/outputs for debugging.
