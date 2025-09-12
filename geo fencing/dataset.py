import json
import random
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# ---------- Config ----------
RISK_SCORE_MAP = {
    "Very High": 20,
    "High": 40,
    "Medium": 70,
    "Standard": 90
}

# ---------- Helpers ----------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def is_inside_circle(user_lat, user_lon, center_lat, center_lon, radius_km):
    return haversine_distance(user_lat, user_lon, center_lat, center_lon) <= radius_km

def assign_risk(user_lat, user_lon, geofences):
    scores, labels = [], []
    for fence in geofences:
        ftype = fence.get("type")
        risk = fence.get("riskLevel")
        score = RISK_SCORE_MAP.get(risk, 100)

        if ftype == "circle":
            center_lat, center_lon = fence["coords"]
            radius = fence["radiusKm"]
            if is_inside_circle(user_lat, user_lon, center_lat, center_lon, radius):
                scores.append(score)
                labels.append(risk)

    if not scores:
        return "Safe", 100
    idx = scores.index(min(scores))
    return labels[idx], scores[idx]

# ---------- Dataset generator ----------
def generate_dataset(geofences, n_samples_per_fence=1000):
    data = []
    for fence in geofences:
        if fence.get("type") == "circle":
            center_lat, center_lon = fence["coords"]
            radius = fence["radiusKm"]

            # Sample uniformly within a square bounding box around the circle
            for _ in range(n_samples_per_fence):
                lat = center_lat + random.uniform(-0.1, 0.1)  # tweak range based on radius
                lon = center_lon + random.uniform(-0.1, 0.1)
                risk_label, score = assign_risk(lat, lon, geofences)
                data.append({
                    "latitude": lat,
                    "longitude": lon,
                    "risk_label": risk_label,
                    "safety_score": score
                })
    return pd.DataFrame(data)

# ---------- Example ----------
if __name__ == "__main__":
    # Load your full JSON file
    with open("geofences.json", "r") as f:
        geofences = json.load(f)

    df = generate_dataset(geofences, n_samples_per_fence=500)
    print(df.head())
    df.to_csv("synthetic_geofence_dataset.csv", index=False)
    print("Dataset saved as synthetic_geofence_dataset.csv")
