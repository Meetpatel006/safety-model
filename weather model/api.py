# Weather Safety Prediction FastAPI Server
# Serves the trained weather safety model via REST API

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Union
import joblib
import pandas as pd
import numpy as np
import logging
import requests
from datetime import datetime, timedelta
import uvicorn
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WeatherSafetyPredictor:
    """Weather Safety Prediction Model Wrapper Class"""

    def __init__(self, model_path: str = "weather_safety_model_kaggle.pkl"):
        """Initialize the predictor with the trained model"""
        # Resolve model path relative to this file's directory for robustness
        base_dir = Path(__file__).resolve().parent
        resolved_path = Path(model_path)
        if not resolved_path.is_absolute():
            resolved_path = base_dir / model_path

        self.model_path = str(resolved_path)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._load_model()

    def _load_model(self) -> bool:
        """Load the trained model and preprocessing objects"""
        logger.info(f"Attempting to load model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            return False

        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']

            logger.info("Model loaded successfully")
            logger.info(f"Features expected: {len(self.feature_names)}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def prepare_features(self, weather_data: dict) -> pd.DataFrame:
        """Prepare features for prediction in the same format as training"""
        df = pd.DataFrame([weather_data])

        # Create derived features that were created during training
        if 'temperature' in df.columns and 'apparent_temperature' in df.columns:
            df['temp_difference'] = df['apparent_temperature'] - df['temperature']

        if 'temperature' in df.columns and 'humidity' in df.columns:
            # Heat index approximation
            df['heat_index'] = df['temperature'] + 0.5 * (df['humidity'] * 100 - 50)

        if 'wind_speed' in df.columns and 'temperature' in df.columns:
            # Wind chill approximation (for temperatures below 10°C)
            df['wind_chill'] = np.where(
                df['temperature'] < 10,
                13.12 + 0.6215 * df['temperature'] - 11.37 * (df['wind_speed'] ** 0.16) + 0.3965 * df['temperature'] * (df['wind_speed'] ** 0.16),
                df['temperature']
            )

        # Create interaction features
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']

        if 'wind_speed' in df.columns and 'visibility' in df.columns:
            df['wind_visibility_interaction'] = df['wind_speed'] / (df['visibility'] + 1)

        if 'pressure' in df.columns:
            df['pressure_deviation'] = abs(df['pressure'] - 1013.25)

        return df

    def fetch_openmeteo_data(self, latitude: float, longitude: float, days: int = 7) -> dict:
        """Fetch weather data from Open-Meteo API"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                "wind_direction_10m", "pressure_msl", "precipitation",
                "visibility", "apparent_temperature"
            ]
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def predict(self, weather_data: Union[dict, 'WeatherInput']) -> dict:
        """Predict weather safety score from input weather conditions"""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not loaded")

        try:
            # Convert input to dictionary if it's a WeatherInput object
            if hasattr(weather_data, 'dict'):
                weather_dict = weather_data.dict()
            else:
                weather_dict = weather_data

            # Prepare features
            feature_df = self.prepare_features(weather_dict)

            # Ensure all required features are present
            missing_features = []
            for feature in self.feature_names:
                if feature not in feature_df.columns:
                    feature_df[feature] = 0  # Default value for missing features
                    missing_features.append(feature)

            # Select features in the correct order
            X = feature_df[self.feature_names]

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]

            safety_labels = {
                0: "Very Safe",
                1: "Safe",
                2: "Moderate Risk",
                3: "High Risk",
                4: "Very High Risk"
            }

            result = {
                "safety_score": int(prediction),
                # Map model class probabilities to a 1-100 safety score.
                # We'll use a weighted mapping where classes 0-4 map to ranges across 1-100.
                # Compute expected class index (probability-weighted) then scale to 1-100.
                "safety_score_100": int(max(1, min(100, round((np.dot(np.arange(len(probabilities)), probabilities) / (len(probabilities)-1)) * 99 + 1)))),
                "safety_category": safety_labels[prediction],
                "confidence": float(probabilities.max()),
                "probabilities": {
                    safety_labels[i]: float(prob)
                    for i, prob in enumerate(probabilities)
                },
                "input_features": {col: float(val) for col, val in X.iloc[0].to_dict().items()}
            }

            if missing_features:
                logger.warning(f"Missing features filled with defaults: {missing_features}")

            return result

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_from_openmeteo(self, latitude: float, longitude: float, days: int = 7) -> dict:
        """Fetch weather data from Open-Meteo and predict safety score"""
        try:
            # Fetch weather data
            weather_data = self.fetch_openmeteo_data(latitude, longitude, days)

            # Extract current weather conditions (first hourly entry)
            hourly = weather_data['hourly']
            current_idx = 0  # Use first available data point

            # Map Open-Meteo data to our model format
            weather_input = {
                "temperature": hourly['temperature_2m'][current_idx],
                "apparent_temperature": hourly['apparent_temperature'][current_idx],
                "humidity": hourly['relative_humidity_2m'][current_idx] / 100,  # Convert to 0-1
                "wind_speed": hourly['wind_speed_10m'][current_idx],
                "wind_bearing": hourly['wind_direction_10m'][current_idx],
                # Convert meters to km; do not arbitrarily cap visibility here so callers can supply higher values
                "visibility": hourly.get('visibility', [10])[current_idx] / 1000,
                "pressure": hourly['pressure_msl'][current_idx],
                "cloud_cover": 0.5,  # Default value
                "precip_rain": 1 if hourly['precipitation'][current_idx] > 0 else 0,
            }

            # Make prediction
            prediction_result = self.predict(weather_input)

            # Add metadata
            prediction_result["input_features"].update({
                "latitude": latitude,
                "longitude": longitude,
                "data_source": "Open-Meteo API",
                "fetch_time": datetime.now().isoformat()
            })

            return prediction_result

        except requests.RequestException as e:
            logger.error(f"Open-Meteo API error: {e}")
            raise RuntimeError(f"Failed to fetch weather data from Open-Meteo: {str(e)}")
        except Exception as e:
            logger.error(f"OpenMeteo prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        return {
            "model_type": type(self.model).__name__,
            "features_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "safety_categories": {
                0: "Very Safe",
                1: "Safe",
                2: "Moderate Risk",
                3: "High Risk",
                4: "Very High Risk"
            }
        }

# Initialize FastAPI app
app = FastAPI(
    title="Weather Safety Prediction API",
    description="API for predicting weather safety scores based on meteorological conditions",
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

class WeatherInput(BaseModel):
    """Input schema for weather data prediction"""
    temperature: float = Field(..., description="Temperature in Celsius", ge=-50, le=60)
    apparent_temperature: Optional[float] = Field(None, description="Apparent temperature in Celsius", ge=-60, le=70)
    humidity: float = Field(..., description="Humidity (0-1 decimal or 0-100 percentage)", ge=0, le=100)
    wind_speed: float = Field(..., description="Wind speed in km/h", ge=0, le=200)
    wind_bearing: Optional[float] = Field(None, description="Wind direction in degrees", ge=0, le=360)
    # Allow visibility and cloud_cover without an arbitrary upper limit; still enforce non-negative
    visibility: float = Field(..., description="Visibility in km", ge=0)
    cloud_cover: Optional[float] = Field(None, description="Cloud cover (0-1)", ge=0)
    pressure: float = Field(..., description="Atmospheric pressure in millibars", ge=900, le=1100)

    @validator('humidity')
    def validate_humidity(cls, v):
        """Normalize humidity to 0-1 range if provided as percentage"""
        if v > 1:
            return v / 100  # Convert percentage to decimal
        return v

    @validator('apparent_temperature', pre=True, always=True)
    def set_apparent_temperature(cls, v, values):
        """Set apparent temperature to temperature if not provided"""
        if v is None and 'temperature' in values:
            return values['temperature']
        return v

class OpenMeteoInput(BaseModel):
    """Input schema for fetching data from Open-Meteo API"""
    latitude: float = Field(..., description="Latitude in decimal degrees", ge=-90, le=90)
    longitude: float = Field(..., description="Longitude in decimal degrees", ge=-180, le=180)
    days: Optional[int] = Field(7, description="Number of days to forecast", ge=1, le=16)

class SafetyPrediction(BaseModel):
    """Output schema for safety predictions"""
    safety_score: int = Field(..., description="Safety score (0-4)")
    safety_score_100: int = Field(..., description="Safety score scaled 1-100")
    safety_category: str = Field(..., description="Safety category name")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probability for each safety category")
    input_features: Dict[str, float] = Field(..., description="Processed input features")

@app.on_event("startup")
def startup_event():
    """FastAPI startup event: initialize the predictor"""
    global predictor
    predictor = WeatherSafetyPredictor()
    if predictor.model is None:
        logger.warning("Model failed to load during startup. Ensure 'weather_safety_model_kaggle.pkl' exists in the same directory as 'api.py' and contains 'model', 'scaler', and 'feature_names'.")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Weather Safety Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "model_loaded": predictor is not None and predictor.model is not None,
        "endpoints": {
            "predict": "/predict",
            "predict_openmeteo": "/predict/openmeteo",
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
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return predictor.get_model_info()

@app.post("/predict", response_model=SafetyPrediction)
async def predict_safety(weather: WeatherInput):
    """Predict weather safety score from input weather conditions"""
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return predictor.predict(weather)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/openmeteo")
async def predict_from_openmeteo(location: OpenMeteoInput):
    """Fetch weather data from Open-Meteo and predict safety score"""
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return predictor.predict_from_openmeteo(location.latitude, location.longitude, location.days)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(weather_list: List[WeatherInput]):
    """Predict safety scores for multiple weather conditions"""
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(weather_list) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")

    try:
        results = []
        for weather in weather_list:
            result = predictor.predict(weather)
            results.append(result)

        return {
            "batch_size": len(results),
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        }

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

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
