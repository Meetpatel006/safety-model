# Weather Safety API Usage Examples
# Run these examples after starting the FastAPI server

import requests
import json

# API base URL (change if deployed elsewhere)
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("=== Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_model_info():
    """Get model information"""
    print("=== Model Info ===")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        model_info = response.json()
        print(f"Model Type: {model_info['model_type']}")
        print(f"Features Count: {model_info['features_count']}")
        print(f"Safety Categories: {model_info['safety_categories']}")
    else:
        print(f"Error: {response.json()}")
    print()

def test_basic_prediction():
    """Test basic weather prediction"""
    print("=== Basic Prediction ===")
    
    # Example: Pleasant weather
    weather_data = {
        "temperature": 22.0,
        "apparent_temperature": 25.0,
        "humidity": 0.6,  # 60%
        "wind_speed": 8.0,  # km/h
        "wind_bearing": 180.0,
        "visibility": 15.0,  # km
        "cloud_cover": 0.3,
        "pressure": 1013.0,
        "precip_rain": 0,
        "precip_snow": 0,
        "summary_clear": 1
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=weather_data)
    print(f"Input: Pleasant weather (22°C, 60% humidity, light wind)")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Safety Score: {result['safety_score']}")
        print(f"Safety Category: {result['safety_category']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Probabilities: {result['probabilities']}")
    else:
        print(f"Error: {response.json()}")
    print()

def test_extreme_weather():
    """Test prediction with extreme weather conditions"""
    print("=== Extreme Weather Prediction ===")
    
    # Example: Dangerous conditions
    weather_data = {
        "temperature": -10.0,  # Very cold
        "apparent_temperature": -15.0,
        "humidity": 0.95,  # Very humid
        "wind_speed": 80.0,  # Very strong wind
        "wind_bearing": 270.0,
        "visibility": 0.5,  # Very poor visibility
        "cloud_cover": 1.0,  # Completely cloudy
        "pressure": 975.0,  # Low pressure (storm)
        "precip_snow": 1,  # Snowing
        "summary_fog": 1,  # Foggy
        "summary_wind": 1,  # Windy
        "daily_summary_snow": 1
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=weather_data)
    print(f"Input: Extreme weather (-10°C, 95% humidity, 80km/h wind, 0.5km visibility, snow)")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Safety Score: {result['safety_score']}")
        print(f"Safety Category: {result['safety_category']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Probabilities: {result['probabilities']}")
    else:
        print(f"Error: {response.json()}")
    print()

def test_openmeteo_integration():
    """Test Open-Meteo API integration"""
    print("=== Open-Meteo Integration ===")
    
    # Example: Weather for New York City
    location_data = {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "days": 3
    }
    
    response = requests.post(f"{BASE_URL}/predict/openmeteo", json=location_data)
    print(f"Input: NYC coordinates (40.7128, -74.0060)")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Safety Score: {result['safety_score']}")
        print(f"Safety Category: {result['safety_category']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Data Source: {result['input_features']['data_source']}")
        print(f"Temperature: {result['input_features']['temperature']:.1f}°C")
        print(f"Humidity: {result['input_features']['humidity']:.2f}")
    else:
        print(f"Error: {response.json()}")
    print()

def test_batch_prediction():
    """Test batch prediction with multiple weather conditions"""
    print("=== Batch Prediction ===")
    
    weather_batch = [
        {
            "temperature": 25.0,
            "humidity": 0.5,
            "wind_speed": 5.0,
            "wind_bearing": 180.0,
            "visibility": 20.0,
            "cloud_cover": 0.1,
            "pressure": 1020.0,
            "summary_clear": 1
        },
        {
            "temperature": 5.0,
            "humidity": 0.8,
            "wind_speed": 35.0,
            "wind_bearing": 270.0,
            "visibility": 8.0,
            "cloud_cover": 0.9,
            "pressure": 990.0,
            "precip_rain": 1,
            "summary_rain": 1
        },
        {
            "temperature": -5.0,
            "humidity": 0.9,
            "wind_speed": 60.0,
            "wind_bearing": 0.0,
            "visibility": 2.0,
            "cloud_cover": 1.0,
            "pressure": 980.0,
            "precip_snow": 1,
            "summary_snow": 1,
            "summary_fog": 1
        }
    ]
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=weather_batch)
    print(f"Input: Batch of 3 weather conditions")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Batch Size: {result['batch_size']}")
        for i, prediction in enumerate(result['predictions']):
            print(f"  Prediction {i+1}: {prediction['safety_category']} (Score: {prediction['safety_score']}, Confidence: {prediction['confidence']:.3f})")
    else:
        print(f"Error: {response.json()}")
    print()

def test_input_validation():
    """Test input validation with invalid data"""
    print("=== Input Validation Test ===")
    
    # Invalid temperature (too high)
    invalid_data = {
        "temperature": 100.0,  # Too high
        "humidity": 0.6,
        "wind_speed": 10.0,
        "wind_bearing": 180.0,
        "visibility": 15.0,
        "pressure": 1013.0
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
    print(f"Input: Invalid temperature (100°C)")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def main():
    """Run all tests"""
    print("Weather Safety API Testing")
    print("=" * 50)
    print()
    
    try:
        test_health_check()
        test_model_info()
        test_basic_prediction()
        test_extreme_weather()
        test_openmeteo_integration()
        test_batch_prediction()
        test_input_validation()
        
        print("All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the API server.")
        print("Make sure the FastAPI server is running on localhost:8000")
        print("Run: python main.py  # or uvicorn main:app --reload")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()