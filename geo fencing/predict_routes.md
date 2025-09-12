(The file `c:\Users\hites\Desktop\Coding\smart-safety-model\geo fencing\predict_routes.md` exists, but is empty)
## GeoFence - Predict Routes (API + Usage)

This document explains how to run and use the GeoFence safety prediction API included in this folder. It covers the purpose of each file, the input/output formats, feature extraction logic, examples (curl and Python), running locally, and troubleshooting notes.

### Overview

The GeoFence Safety API predicts a categorical risk label for a given geographic coordinate (latitude, longitude) and exposes a simple HTTP endpoint (`/predict`). The endpoint returns a predicted risk label plus a normalized safety score compatible with other models in the project.

The model and supporting artifacts are included in this folder:

- `geofence_safety_model.pkl` - Trained classification model (joblib format)
- `label_encoder.pkl` - Label encoder used to convert labels to/from numeric encoding
- `geofences.json` - Geofence definitions used for feature extraction (circle fences expected)
- `geofence_features.csv` - (optional) features used during training
- `synthetic_geofence_dataset.csv` - (optional) synthetic dataset used in development/training
- `api.py` - FastAPI application that provides the `/predict` endpoint
- `dataset.py`, `feature.py`, `training.py` - training & feature code used to build the model

### Contract (inputs / outputs / error modes)

- Input: JSON object with two floats: `latitude` and `longitude`.
- Output: JSON object with original coordinates, predicted categorical risk label, and two forms of a safety score:
	- `predicted_risk_label`: categorical label (e.g., "Safe", "Medium", "High")
	- `predicted_safety_score`: mapped score from RISK_SCORE_MAP (documented below)
	- `safety_score_100`: clamped 1-100 integer for compatibility with other systems
- Error modes: FastAPI will return 422 for invalid/malformed input; other errors (missing model files, I/O) return 500.

Example request body:

```
{
	"latitude": 37.7749,
	"longitude": -122.4194
}
```

Example response (success):

```
{
	"latitude": 37.7749,
	"longitude": -122.4194,
	"predicted_risk_label": "Medium",
	"predicted_safety_score": 70,
	"safety_score_100": 70
}
```

### Feature extraction summary (how prediction is computed)

The `api.py` code performs a small feature extraction step before sending data to the model. In short:

- It loads `geofences.json` and iterates over fences of type `circle`.
- For each circle fence the API computes the Haversine distance between the query point and the fence center (in kilometers).
- It collects: the minimum distance to any circular geofence, the risk level of the nearest geofence, and whether the point is inside a fence radius.
- The final feature vector passed to the model is: `[latitude, longitude, min_distance_km, inside_flag, mapped_risk_score]` where `mapped_risk_score` is taken from a small map defined in the API.

RISK_SCORE_MAP used by the API (maps label -> numeric score):

- "Very High": 20
- "High": 40
- "Medium": 70
- "Standard": 90
- "Safe": 100

Notes:
- The API produces a `safety_score_100` by clamping the mapped score to [1,100] for compatibility with other projects.
- The API's feature extraction was designed to match what the training pipeline expects; keep the feature order when calling the model.

### Running locally (development)

Prerequisites:
- Python 3.8+ (the code has been used with Python 3.10+)
- `pip install -r requirements.txt` (ensure `fastapi`, `uvicorn`, and `joblib` are installed)

Start the API using Uvicorn (development mode):

```powershell
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

After startup, the OpenAPI docs will be available at:

http://127.0.0.1:8000/docs

And the endpoint is:

POST http://127.0.0.1:8000/predict

If you prefer to bind to all interfaces (for use in containers or remote testing), change host to `0.0.0.0`.

### Example clients

Curl (Linux/macOS) example:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"latitude":37.7749, "longitude":-122.4194}'
```

PowerShell (using Invoke-RestMethod):

```powershell
$body = @{ latitude = 37.7749; longitude = -122.4194 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -Body $body -ContentType 'application/json'
```

Python example (requests):

```python
import requests

url = "http://127.0.0.1:8000/predict"
payload = {"latitude": 37.7749, "longitude": -122.4194}
resp = requests.post(url, json=payload)
print(resp.json())
```

### Inspect / modify geofences

`geofences.json` holds the geofence definitions. Each fence object commonly contains these keys (example shape):

```
{
	"id": "fence-1",
	"type": "circle",
	"coords": [37.7749, -122.4194],
	"radiusKm": 1.5,
	"riskLevel": "High"
}
```

When adding or editing geofences, validate:
- `coords` are [latitude, longitude]
- `radiusKm` is a numeric radius in kilometers
- `riskLevel` is one of the levels your pipeline expects (e.g., "Very High", "High", "Medium", "Standard", "Safe")

After changing geofences, restart the API to pick up changes.

### Troubleshooting & common issues

- Missing model or encoder files: If `geofence_safety_model.pkl` or `label_encoder.pkl` are not present or corrupted, the API will fail to import at startup. Ensure both files exist in the same folder as `api.py`.
- Model / feature mismatch: If you train a new model with a different feature order or different preprocessing, update `api.py`'s `extract_features` to match training.
- Distance / inside calculation: The current extraction finds the nearest distance and uses a boolean `inside` flag. If you add non-circular geofences, the current `extract_features` may ignore them or require enhancement.
- Permissions / path problems on Windows: When running on Windows, ensure your working directory is the `geo fencing` folder (or use absolute paths when loading the `.pkl` and `.json` files).

Known implementation note:
- The `extract_features` helper in `api.py` uses the last iterated `fence` variable when calculating the `inside` flag — if you modify feature extraction be careful to compute `inside` using the fence that corresponds to the minimum distance.

### Testing locally

1. Start the API as shown above.
2. Use the Python example or Curl to send a few coordinates.
3. Confirm the returned `predicted_risk_label` matches expectations for points inside and outside geofences.

### Extending or integrating

- Batch predictions: For many points, wrap multiple HTTP requests or extend the API to accept arrays of coordinates. Make sure to batch feature extraction to reuse geofence computations efficiently.
- Logging: Add structured logging to `api.py` to record incoming requests and predictions for auditing.
- Metrics: Add a small Prometheus exporter or endpoint that reports request latency and prediction counts.

### Quick checklist before production

- [ ] Ensure the model was trained with production-like geofences and features
- [ ] Pin `requirements.txt` to exact package versions used in development
- [ ] Add input validation and rate limiting if exposing the endpoint publicly
- [ ] Add tests: unit tests for `extract_features` and an integration test that hits `/predict` using a test client

### References

- Source: `api.py` (this folder)
- Geofence definitions: `geofences.json`
- Training code: `training.py`, `feature.py`, `dataset.py`

---

If you'd like, I can also:

- Add a small unit test that validates `extract_features` for a few edge cases.
- Add a batch endpoint that accepts an array of points and returns predictions for each.

