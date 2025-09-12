# api.py
import joblib
import json
from fastapi import FastAPI
from pydantic import BaseModel
from math import radians, sin, cos, sqrt, atan2

# ---------- Config ----------
RISK_SCORE_MAP = {
    "Very High": 20,
    "High": 40,
    "Medium": 70,
    "Standard": 90,
    "Safe": 100
}

# ---------- Load model + encoder ----------
model = joblib.load("geofence_safety_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load geofences
with open("geofences.json", "r") as f:
    geofences = json.load(f)

# ---------- Helpers ----------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def extract_features(lat, lon, geofences):
    distances, risks = [], []
    for fence in geofences:
        if fence.get("type") == "circle":
            center_lat, center_lon = fence["coords"]
            radius = fence["radiusKm"]
            dist = haversine_distance(lat, lon, center_lat, center_lon)
            distances.append(dist)
            risks.append(fence.get("riskLevel"))

    if distances:
        min_d = min(distances)
        min_idx = distances.index(min_d)
        closest_risk = risks[min_idx]
        inside = 1 if min_d <= fence["radiusKm"] else 0
    else:
        min_d, closest_risk, inside = 9999, "Safe", 0

    return [lat, lon, min_d, inside, RISK_SCORE_MAP.get(closest_risk, 100)]

# ---------- FastAPI ----------
app = FastAPI(title="GeoFence Safety Score API")

class Location(BaseModel):
    latitude: float
    longitude: float

@app.post("/predict")
def predict_safety(loc: Location):
    features = extract_features(loc.latitude, loc.longitude, geofences)
    pred_encoded = model.predict([features])[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    return {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "predicted_risk_label": pred_label,
        "predicted_safety_score": RISK_SCORE_MAP.get(pred_label, 100),
        # Provide a 1-100 safety score for compatibility with other models.
        # The existing RISK_SCORE_MAP already uses values in 0-100-ish range; clamp to [1,100].
        "safety_score_100": int(max(1, min(100, RISK_SCORE_MAP.get(pred_label, 100))))
    }
