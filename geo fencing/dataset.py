# FILE: dataset.py (with rounding)

import json
import random
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# (All the config and helper functions remain the same...)
# ---------- Config ----------
RISK_RANGES = {
    "Very High": (0, 20), "High": (21, 40), "Medium": (41, 70),
    "Standard": (71, 90), "Safe": (91, 100)
}
SAFE_TRANSITION_KM = 1.0 

# ---------- Helpers ----------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# ---------- CONTINUOUS SCORING LOGIC (with rounding added) ----------
def assign_risk_continuous(user_lat, user_lon, geofences):
    potential_scores = []
    for fence in geofences:
        if fence.get("type") == "circle":
            center_lat, center_lon = fence["coords"]
            radius = fence["radiusKm"]
            risk_label = fence.get("riskLevel")
            min_score, max_score = RISK_RANGES.get(risk_label, (91, 100))
            dist = haversine_distance(user_lat, user_lon, center_lat, center_lon)

            if dist <= radius:
                score_range = max_score - min_score
                score = min_score + (dist / radius) * score_range
                potential_scores.append(score)
            elif dist <= radius + SAFE_TRANSITION_KM:
                dist_from_edge = dist - radius
                score_range = 100 - max_score
                score = max_score + (dist_from_edge / SAFE_TRANSITION_KM) * score_range
                potential_scores.append(score)

    if not potential_scores:
        return "Safe", 100.0

    min_final_score = min(potential_scores)
    final_label = "Safe"
    for label, (min_s, max_s) in RISK_RANGES.items():
        if min_s <= round(min_final_score) <= max_s:
            final_label = label
            break

    # === THIS IS THE ONLY LINE THAT CHANGED ===
    # Round the final score to 2 decimal places for cleanliness.
    return final_label, round(min_final_score, 2)

# (The generate_dataset function and main block remain the same...)
def generate_dataset(geofences, n_samples_per_fence=1000):
    data = []
    for fence in geofences:
        if fence.get("type") == "circle":
            center_lat, center_lon = fence["coords"]
            for _ in range(n_samples_per_fence):
                lat = center_lat + random.uniform(-0.1, 0.1)
                lon = center_lon + random.uniform(-0.1, 0.1)
                risk_label, score = assign_risk_continuous(lat, lon, geofences)
                data.append({
                    "latitude": lat, "longitude": lon, "risk_label": risk_label,
                    "safety_score": score
                })
    return pd.DataFrame(data)

if __name__ == "__main__":
    with open("geofences.json", "r") as f:
        geofences = json.load(f)
    df = generate_dataset(geofences, n_samples_per_fence=500)
    print("Generated Dataset with ROUNDED Continuous Scores:")
    print(df.head())
    df.to_csv("synthetic_geofence_dataset.csv", index=False)
    print("\nRounded continuous score dataset saved.")