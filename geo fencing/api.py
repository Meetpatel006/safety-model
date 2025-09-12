# GeoFence Safety Prediction API
# Serves the trained geofence safety model via REST API

import joblib
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from math import radians, sin, cos, sqrt, atan2
from typing import Optional, Dict
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
                 encoder_path: str = "label_encoder.pkl",
                 geofences_path: str = "geofences.json"):
        """Initialize the predictor with the trained model and geofence data"""
        # Resolve paths relative to this file's directory for robustness
        base_dir = Path(__file__).resolve().parent

        self.model_path = str(base_dir / model_path)
        self.encoder_path = str(base_dir / encoder_path)
        self.geofences_path = str(base_dir / geofences_path)

        self.model = None
        self.label_encoder = None
        self.geofences = None
        self.risk_score_map = {
            "Very High": 20,
            "High": 40,
            "Medium": 70,
            "Standard": 90,
            "Safe": 100
        }

        self._load_model_and_data()

    def _load_model_and_data(self) -> bool:
        """Load the trained model, encoder, and geofence data"""
        logger.info("Loading GeoFence safety model and data...")

        try:
            # Load model
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info("Model loaded successfully")
            else:
                logger.error(f"Model file not found: {self.model_path}")
                return False

            # Load label encoder
            if os.path.exists(self.encoder_path):
                self.label_encoder = joblib.load(self.encoder_path)
                logger.info("Label encoder loaded successfully")
            else:
                logger.error(f"Label encoder file not found: {self.encoder_path}")
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
        """Calculate the great circle distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    def extract_features(self, lat: float, lon: float) -> list:
        """Extract features for the given location based on geofence data"""
        distances, risks = [], []

        for fence in self.geofences:
            if fence.get("type") == "circle":
                center_lat, center_lon = fence["coords"]
                radius = fence["radiusKm"]
                dist = self.haversine_distance(lat, lon, center_lat, center_lon)
                distances.append(dist)
                risks.append(fence.get("riskLevel"))

        if distances:
            min_d = min(distances)
            min_idx = distances.index(min_d)
            closest_risk = risks[min_idx]
            # Use the radius from the closest fence for inside check
            closest_fence = self.geofences[min_idx]
            inside = 1 if min_d <= closest_fence["radiusKm"] else 0
        else:
            min_d, closest_risk, inside = 9999, "Safe", 0

        return [lat, lon, min_d, inside, self.risk_score_map.get(closest_risk, 100)]

    def predict(self, latitude: float, longitude: float) -> dict:
        """Predict safety score for the given location"""
        if self.model is None or self.label_encoder is None or self.geofences is None:
            raise RuntimeError("Model or data not loaded")

        try:
            # Extract features
            features = self.extract_features(latitude, longitude)

            # Make prediction
            pred_encoded = self.model.predict([features])[0]
            pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]

            # Calculate safety score on 1-100 scale
            safety_score_100 = int(max(1, min(100, self.risk_score_map.get(pred_label, 100))))

            return {
                "latitude": latitude,
                "longitude": longitude,
                "predicted_risk_label": pred_label,
                "predicted_safety_score": self.risk_score_map.get(pred_label, 100),
                "safety_score_100": safety_score_100,
                "features": {
                    "distance_to_nearest_geofence": features[2],
                    "inside_geofence": bool(features[3]),
                    "geofence_risk_level": features[4]
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def get_model_info(self) -> dict:
        """Get information about the loaded model and data"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        return {
            "model_type": type(self.model).__name__,
            "geofences_count": len(self.geofences) if self.geofences else 0,
            "risk_categories": list(self.risk_score_map.keys()),
            "risk_score_mapping": self.risk_score_map
        }

# Initialize FastAPI app
app = FastAPI(
    title="GeoFence Safety Prediction API",
    description="API for predicting geofence safety scores based on location data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

class Location(BaseModel):
    """Input schema for location data prediction"""
    latitude: float = Field(..., description="Latitude in decimal degrees", ge=-90, le=90)
    longitude: float = Field(..., description="Longitude in decimal degrees", ge=-180, le=180)

class SafetyPrediction(BaseModel):
    """Output schema for safety predictions"""
    latitude: float = Field(..., description="Input latitude")
    longitude: float = Field(..., description="Input longitude")
    predicted_risk_label: str = Field(..., description="Predicted risk category")
    predicted_safety_score: int = Field(..., description="Safety score (20-100)")
    safety_score_100: int = Field(..., description="Safety score scaled 1-100")
    features: Dict = Field(..., description="Extracted features used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")

@app.on_event("startup")
def startup_event():
    """FastAPI startup event: initialize the predictor"""
    global predictor
    predictor = GeoFenceSafetyPredictor()
    if predictor.model is None:
        logger.warning("Model failed to load during startup. Ensure required files exist in the geo fencing directory.")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "GeoFence Safety Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "model_loaded": predictor is not None and predictor.model is not None,
        "endpoints": {
            "predict": "/predict",
            "model_info": "/model/info",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = predictor is not None and predictor.model is not None
    return {
        "status": "healthy" if model_status else "unhealthy",
        "model_loaded": model_status,
        "geofences_loaded": predictor is not None and predictor.geofences is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return predictor.get_model_info()

@app.post("/predict", response_model=SafetyPrediction)
async def predict_safety(location: Location):
    """Predict geofence safety score for the given location"""
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return predictor.predict(location.latitude, location.longitude)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"message": f"Invalid input: {str(exc)}"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )
