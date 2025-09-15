# FILE: api.py
# --- FINAL VERSION FOR REGRESSION MODEL ---

import joblib
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from math import radians, sin, cos, sqrt, atan2
from typing import Dict
from datetime import datetime
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeoFenceSafetyPredictor:
    """GeoFence Safety Prediction Model Wrapper Class"""

    def __init__(self, model_path: str = "geofence_safety_model.pkl",
                 geofences_path: str = "geofences.json"):
        """Initialize the predictor with the trained REGRESSION model and geofence data"""
        base_dir = Path(__file__).resolve().parent

        self.model_path = str(base_dir / model_path)
        self.geofences_path = str(base_dir / geofences_path)

        self.model = None
        self.geofences = None
        # This map is now used for both feature extraction and final labeling
        self.risk_score_map = {
            "Very High": 20, "High": 40, "Medium": 70,
            "Standard": 90, "Safe": 100
        }

        self._load_model_and_data()

    def _load_model_and_data(self) -> bool:
        """Load the trained model and geofence data"""
        logger.info("Loading GeoFence safety model and data...")
        try:
            # Load the regression model
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Regression model ({type(self.model).__name__}) loaded successfully")
            else:
                logger.error(f"Model file not found: {self.model_path}")
                return False

            # Load geofences
            if os.path.exists(self.geofences_path):
                with open(self.geofences_path, "r") as f:
                    self.geofences = json.load(f)
                logger.info(f"Geofences loaded successfully: {len(self.geofences)} geofences")
            else:
                logger.error(f"Geofences file not found: {self.geofences_path}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error loading model/data: {e}")
            return False

    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points"""
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    def extract_features_for_info(self, lat: float, lon: float) -> list:
        """Extracts contextual features to return to the user. Not for model prediction."""
        distances, risks, radii = [], [], []
        for fence in self.geofences:
            if fence.get("type") == "circle":
                center_lat, center_lon = fence["coords"]
                dist = self.haversine_distance(lat, lon, center_lat, center_lon)
                distances.append(dist)
                risks.append(fence.get("riskLevel"))
                radii.append(fence["radiusKm"])
        
        if distances:
            min_d = min(distances)
            min_idx = distances.index(min_d)
            closest_risk = risks[min_idx]
            inside = 1 if min_d <= radii[min_idx] else 0
        else:
            min_d, closest_risk, inside = 9999, "Safe", 0
            
        return [min_d, inside, self.risk_score_map.get(closest_risk, 100)]
    
    def _score_to_label(self, score: float) -> str:
        """Converts a numerical safety score back to a risk label."""
        score = int(score) # Convert to integer for simple range checking
        if 0 <= score <= 20: return "Very High"
        if 21 <= score <= 40: return "High"
        if 41 <= score <= 70: return "Medium"
        if 71 <= score <= 90: return "Standard"
        return "Safe"

    def predict(self, latitude: float, longitude: float) -> dict:
        """Predict safety score for the given location using the REGRESSION model."""
        if self.model is None or self.geofences is None:
            raise RuntimeError("Model or data not loaded")

        try:
            # The model was trained on ONLY latitude and longitude
            model_features = [[latitude, longitude]]

            # The regression model predicts a continuous score
            predicted_score = self.model.predict(model_features)[0]
            
            # Convert the score to a label using our helper function
            pred_label = self._score_to_label(predicted_score)
            
            # Extract other features just for user information
            info_features = self.extract_features_for_info(latitude, longitude)

            return {
                "latitude": latitude,
                "longitude": longitude,
                "predicted_risk_label": pred_label,
                "predicted_safety_score": round(predicted_score, 2),
                "safety_score_100": int(max(1, min(100, predicted_score))),
                "features": {
                    "distance_to_nearest_geofence_km": round(info_features[0], 2),
                    "is_inside_geofence": bool(info_features[1]),
                    "risk_score_of_nearest_geofence": info_features[2]
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def get_model_info(self) -> dict:
        """Get information about the loaded model and data"""
        if self.model is None: raise RuntimeError("Model not loaded")
        return {
            "model_type": type(self.model).__name__,
            "geofences_count": len(self.geofences) if self.geofences else 0,
            "model_features": self.model.feature_names_in_.tolist() if hasattr(self.model, 'feature_names_in_') else ["latitude", "longitude"]
        }

# --- FastAPI App Setup ---
app = FastAPI(
    title="GeoFence Safety Prediction API",
    description="API for predicting geofence safety scores based on location data using a regression model.",
    version="2.0.0", # Version bumped to reflect regression model
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = None

class Location(BaseModel):
    latitude: float = Field(..., description="Latitude in decimal degrees", ge=-90, le=90)
    longitude: float = Field(..., description="Longitude in decimal degrees", ge=-180, le=180)

class SafetyPrediction(BaseModel):
    latitude: float
    longitude: float
    predicted_risk_label: str
    predicted_safety_score: float
    safety_score_100: int
    features: Dict
    timestamp: str

@app.on_event("startup")
def startup_event():
    global predictor
    predictor = GeoFenceSafetyPredictor()
    if predictor.model is None:
        logger.warning("Model failed to load during startup. Ensure 'geofence_safety_model.pkl' and 'geofences.json' exist.")

@app.get("/")
async def root():
    return {
        "message": "GeoFence Safety Prediction API",
        "version": "2.0.0",
        "docs": "/docs",
        "model_loaded": predictor is not None and predictor.model is not None
    }

@app.get("/health")
async def health_check():
    model_status = predictor is not None and predictor.model is not None
    return {
        "status": "healthy" if model_status else "unhealthy",
        "model_loaded": model_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def model_info():
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor.get_model_info()

@app.post("/predict", response_model=SafetyPrediction)
async def predict_safety(location: Location):
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return predictor.predict(location.latitude, location.longitude)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred."}
    )