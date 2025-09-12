import json
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# ---------- Config ----------
RISK_SCORE_MAP = {
    "Very High": 20,
    "High": 40,
    "Medium": 70,
    "Standard": 90,
    "Safe": 100
}

# ---------- Helpers ----------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# ---------- Feature Engineering ----------
def add_features(df, geofences):
    """
    Add geospatial features for each sample.
    Features:
      - min_distance_to_geofence
      - inside_any_geofence (binary)
      - closest_geofence_risk (encoded as numeric score)
    """
    min_distances = []
    inside_flags = []
    closest_risks = []

    for idx, row in df.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        distances, risks = [], []

        for fence in geofences:
            if fence.get("type") == "circle":
                center_lat, center_lon = fence["coords"]
                radius = fence["radiusKm"]
                dist = haversine_distance(lat, lon, center_lat, center_lon)
                distances.append(dist)
                risks.append(fence.get("riskLevel"))

        # compute features
        if distances:
            min_d = min(distances)
            min_idx = distances.index(min_d)
            closest_risk = risks[min_idx]
            inside = 1 if min_d <= fence["radiusKm"] else 0
        else:
            min_d = 9999
            closest_risk = "Safe"
            inside = 0

        min_distances.append(min_d)
        inside_flags.append(inside)
        closest_risks.append(RISK_SCORE_MAP.get(closest_risk, 100))

    df["min_distance_to_geofence"] = min_distances
    df["inside_any_geofence"] = inside_flags
    df["closest_geofence_risk_score"] = closest_risks

    return df

# ---------- Example ----------
if __name__ == "__main__":
    # Load dataset generated earlier
    df = pd.read_csv("synthetic_geofence_dataset.csv")

    # Load geofences
    with open("geofences.json", "r") as f:
        geofences = json.load(f)

    df = add_features(df, geofences)
    print(df.head())

    df.to_csv("geofence_features.csv", index=False)
    print("Feature dataset saved as geofence_features.csv")
