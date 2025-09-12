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

# Global model instance
model_instance = None
scaler_instance = None
feature_names = None

class WeatherInput(BaseModel):
    """Input schema for weather data prediction"""
    temperature: float = Field(..., description="Temperature in Celsius", ge=-50, le=60)
    apparent_temperature: Optional[float] = Field(None, description="Apparent temperature in Celsius", ge=-60, le=70)
    humidity: float = Field(..., description="Humidity (0-1 decimal or 0-100 percentage)", ge=0, le=100)
    wind_speed: float = Field(..., description="Wind speed in km/h", ge=0, le=200)
    wind_bearing: Optional[float] = Field(None, description="Wind direction in degrees", ge=0, le=360)
    visibility: float = Field(..., description="Visibility in km", ge=0, le=50)
    cloud_cover: Optional[float] = Field(None, description="Cloud cover (0-1)", ge=0, le=1)
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

def load_model():
    """Load the trained model and preprocessing objects"""
    global model_instance, scaler_instance, feature_names
    
    model_path = "weather_safety_model_kaggle.pkl"
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    try:
        model_data = joblib.load(model_path)
        model_instance = model_data['model']
        scaler_instance = model_data['scaler']
        feature_names = model_data['feature_names']
        
        logger.info("Model loaded successfully")
        logger.info(f"Features expected: {len(feature_names)}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

@app.on_event("startup")
def startup_event():
    """FastAPI startup event: attempt to load the model and log status."""
    loaded = load_model()
    if not loaded:
        logger.warning("Model failed to load during startup. Ensure 'weather_safety_model_kaggle.pkl' exists in the same directory as 'api.py' and contains 'model', 'scaler', and 'feature_names'.")

def prepare_features(weather_data: dict) -> pd.DataFrame:
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

def fetch_openmeteo_data(latitude: float, longitude: float, days: int = 7) -> dict:
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



@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Weather Safety Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "model_loaded": model_instance is not None,
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
    model_status = model_instance is not None
    return {
        "status": "healthy" if model_status else "unhealthy",
        "model_loaded": model_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model_instance).__name__,
        "features_count": len(feature_names),
        "feature_names": feature_names,
        "safety_categories": {
            0: "Very Safe",
            1: "Safe", 
            2: "Moderate Risk",
            3: "High Risk",
            4: "Very High Risk"
        }
    }

@app.post("/predict", response_model=SafetyPrediction)
async def predict_safety(weather: WeatherInput):
    """Predict weather safety score from input weather conditions"""
    if model_instance is None or scaler_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to dictionary
        weather_dict = weather.dict()
        
        # Prepare features
        feature_df = prepare_features(weather_dict)
        
        # Ensure all required features are present
        missing_features = []
        for feature in feature_names:
            if feature not in feature_df.columns:
                feature_df[feature] = 0  # Default value for missing features
                missing_features.append(feature)
        
        # Select features in the correct order
        X = feature_df[feature_names]
        
        # Scale features
        X_scaled = scaler_instance.transform(X)
        
        # Make prediction
        prediction = model_instance.predict(X_scaled)[0]
        probabilities = model_instance.predict_proba(X_scaled)[0]
        
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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/openmeteo")
async def predict_from_openmeteo(location: OpenMeteoInput):
    """Fetch weather data from Open-Meteo and predict safety score"""
    if model_instance is None or scaler_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Fetch weather data
        weather_data = fetch_openmeteo_data(
            location.latitude, 
            location.longitude, 
            location.days
        )
        
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
            "visibility": min(hourly.get('visibility', [10])[current_idx] / 1000, 50),  # Convert meters to km and cap at 50
            "pressure": hourly['pressure_msl'][current_idx],
            "cloud_cover": 0.5,  # Default value
            "precip_rain": 1 if hourly['precipitation'][current_idx] > 0 else 0,
        }
        
        # Create WeatherInput object for validation
        validated_input = WeatherInput(**weather_input)
        
        # Make prediction (predict_safety returns a dict)
        prediction_result = await predict_safety(validated_input)

        # Ensure we have a dict and update its input_features safely
        if isinstance(prediction_result, dict):
            input_feats = prediction_result.get("input_features", {}) or {}
            input_feats["latitude"] = location.latitude
            input_feats["longitude"] = location.longitude
            input_feats["data_source"] = "Open-Meteo API"
            input_feats["fetch_time"] = datetime.now().isoformat()
            prediction_result["input_features"] = input_feats
            return prediction_result
        else:
            # Fallback: try to convert to a dict-like object
            try:
                pr = dict(prediction_result)
                input_feats = pr.get("input_features", {}) or {}
                input_feats["latitude"] = location.latitude
                input_feats["longitude"] = location.longitude
                input_feats["data_source"] = "Open-Meteo API"
                input_feats["fetch_time"] = datetime.now().isoformat()
                pr["input_features"] = input_feats
                return pr
            except Exception:
                logger.error("Unexpected prediction result type when calling predict_safety")
                raise HTTPException(status_code=500, detail="Prediction returned unexpected result type")
        
    except requests.RequestException as e:
        logger.error(f"Open-Meteo API error: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch weather data from Open-Meteo")
    except Exception as e:
        logger.error(f"OpenMeteo prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(weather_list: List[WeatherInput]):
    """Predict safety scores for multiple weather conditions"""
    if model_instance is None or scaler_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(weather_list) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    try:
        results = []
        for weather in weather_list:
            result = await predict_safety(weather)
            results.append(result)
        
        return {
            "batch_size": len(results),
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

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