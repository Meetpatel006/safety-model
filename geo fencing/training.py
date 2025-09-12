import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# ---------- Load Data ----------
df = pd.read_csv("geofence_features.csv")

# Features and target
X = df[["latitude", "longitude", "min_distance_to_geofence",
        "inside_any_geofence", "closest_geofence_risk_score"]]
y = df["risk_label"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ---------- Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---------- Model ----------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ---------- Evaluation ----------
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ---------- Save Model ----------
joblib.dump(model, "geofence_safety_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Model and label encoder saved.")
