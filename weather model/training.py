# Weather Safety Prediction Model for Kaggle Dataset
# Dataset: muthuj7/weather-dataset with specific columns
# This script creates safety labels and trains an ML model

import pandas as pd
import numpy as np
import requests
import logging
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import re
import warnings
warnings.filterwarnings('ignore')
import shutil
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weather_model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WeatherSafetyModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.precip_encoder = LabelEncoder()
        
    def fetch_openmeteo_data(self, latitude=40.7128, longitude=-74.0060, days=30):
        """
        Fetch weather data from Open-Meteo API
        """
        logger.info("Fetching data from Open-Meteo API...")
        
        # Calculate date range
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
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            logger.info("Successfully fetched Open-Meteo data")
            return data
        except Exception as e:
            logger.error(f"Error fetching Open-Meteo data: {e}")
            return None

    def load_kaggle_dataset(self, dataset_path):
        """
        Load the Kaggle weather dataset with specific columns:
        Formatted Date, Summary, Precip Type, Temperature (C), Apparent Temperature (C), 
        Humidity, Wind Speed (km/h), Wind Bearing (degrees), Visibility (km), 
        Loud Cover, Pressure (millibars), Daily Summary
        """
        if dataset_path is None:
            logger.error("Please provide the path to the downloaded Kaggle dataset")
            logger.info("Download from: https://www.kaggle.com/datasets/muthuj7/weather-dataset")
            return None
        
        try:
            # Load the dataset
            df = pd.read_csv(dataset_path)
            logger.info(f"Loaded dataset with shape: {df.shape}")
            logger.info(f"Original columns: {list(df.columns)}")
            
            # Standardize column names for easier processing
            column_mapping = {
                'Formatted Date': 'formatted_date',
                'Summary': 'summary',
                'Precip Type': 'precip_type',
                'Temperature (C)': 'temperature',
                'Apparent Temperature (C)': 'apparent_temperature',
                'Humidity': 'humidity',
                'Wind Speed (km/h)': 'wind_speed',
                'Wind Bearing (degrees)': 'wind_bearing',
                'Visibility (km)': 'visibility',
                'Loud Cover': 'cloud_cover',  # Assuming this is "Cloud Cover"
                'Pressure (millibars)': 'pressure',
                'Daily Summary': 'daily_summary'
            }
            
            # Rename columns if they exist
            df = df.rename(columns=column_mapping)
            
            logger.info(f"Standardized columns: {list(df.columns)}")
            logger.info(f"Dataset info:\n{df.info()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None

    def preprocess_dataset(self, df):
        """
        Preprocess the Kaggle dataset for training
        """
        logger.info("Preprocessing the dataset...")
        
        # Create a copy for processing
        processed_df = df.copy()
        
        # Handle date column
        if 'formatted_date' in processed_df.columns:
            try:
                processed_df['formatted_date'] = pd.to_datetime(processed_df['formatted_date'])
                processed_df['year'] = processed_df['formatted_date'].dt.year
                processed_df['month'] = processed_df['formatted_date'].dt.month
                processed_df['day'] = processed_df['formatted_date'].dt.day
                processed_df['hour'] = processed_df['formatted_date'].dt.hour
                processed_df['day_of_year'] = processed_df['formatted_date'].dt.dayofyear
                logger.info("Date features created successfully")
            except Exception as e:
                logger.warning(f"Error processing date column: {e}")
        
        # Handle precipitation type
        if 'precip_type' in processed_df.columns:
            processed_df['precip_type'] = processed_df['precip_type'].fillna('none')
            # Create binary features for precipitation types
            precip_types = processed_df['precip_type'].unique()
            for precip in precip_types:
                if pd.notna(precip) and precip != 'none':
                    processed_df[f'precip_{precip.lower()}'] = (processed_df['precip_type'] == precip).astype(int)
            logger.info(f"Precipitation types found: {precip_types}")
        
        # Handle summary and daily summary (extract weather conditions)
        for col in ['summary', 'daily_summary']:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].fillna('clear')
                # Extract weather conditions
                processed_df[f'{col}_clear'] = processed_df[col].str.contains('clear|sunny', case=False, na=False).astype(int)
                processed_df[f'{col}_rain'] = processed_df[col].str.contains('rain|drizzle', case=False, na=False).astype(int)
                processed_df[f'{col}_snow'] = processed_df[col].str.contains('snow|sleet', case=False, na=False).astype(int)
                processed_df[f'{col}_cloudy'] = processed_df[col].str.contains('cloud|overcast', case=False, na=False).astype(int)
                processed_df[f'{col}_fog'] = processed_df[col].str.contains('fog|mist', case=False, na=False).astype(int)
                processed_df[f'{col}_wind'] = processed_df[col].str.contains('wind|breezy', case=False, na=False).astype(int)
        
        # Fill missing values for numerical columns
        numerical_cols = ['temperature', 'apparent_temperature', 'humidity', 'wind_speed', 
                         'wind_bearing', 'visibility', 'cloud_cover', 'pressure']
        
        for col in numerical_cols:
            if col in processed_df.columns:
                # Fill missing values with median
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        
        # Create additional weather features
        if 'temperature' in processed_df.columns and 'apparent_temperature' in processed_df.columns:
            processed_df['temp_difference'] = processed_df['apparent_temperature'] - processed_df['temperature']
        
        if 'temperature' in processed_df.columns and 'humidity' in processed_df.columns:
            # Heat index approximation
            processed_df['heat_index'] = processed_df['temperature'] + 0.5 * (processed_df['humidity'] - 50)
        
        if 'wind_speed' in processed_df.columns and 'temperature' in processed_df.columns:
            # Wind chill approximation (for temperatures below 10°C)
            processed_df['wind_chill'] = np.where(
                processed_df['temperature'] < 10,
                13.12 + 0.6215 * processed_df['temperature'] - 11.37 * (processed_df['wind_speed'] ** 0.16) + 0.3965 * processed_df['temperature'] * (processed_df['wind_speed'] ** 0.16),
                processed_df['temperature']
            )
        
        logger.info(f"Preprocessed dataset shape: {processed_df.shape}")
        logger.info("Data preprocessing completed successfully")
        
        return processed_df

    def create_safety_labels(self, df):
        """
        Create safety score labels based on weather conditions
        Safety Score Categories:
        0: Very Safe (Green) - Ideal weather conditions
        1: Safe (Yellow-Green) - Generally safe with minor concerns
        2: Moderate Risk (Yellow) - Some caution advised
        3: High Risk (Orange) - Significant weather hazards
        4: Very High Risk (Red) - Dangerous conditions
        """
        logger.info("Creating safety labels based on weather conditions...")
        
        def calculate_safety_score(row):
            score = 0
            
            # Temperature risk (extreme temperatures are dangerous)
            temp = row.get('temperature', 20)
            if temp < -15 or temp > 45:  # Extreme temperatures
                score += 3
            elif temp < -5 or temp > 40:  # Very hot/cold
                score += 2
            elif temp < 5 or temp > 35:   # Hot/cold
                score += 1
            
            # Apparent temperature (feels-like temperature)
            apparent_temp = row.get('apparent_temperature', temp)
            if abs(apparent_temp - temp) > 10:  # Large difference indicates extreme conditions
                score += 1
            
            # Wind speed risk (convert km/h to more intuitive scale)
            wind = row.get('wind_speed', 0)
            if wind > 70:      # Severe wind (19+ m/s)
                score += 3
            elif wind > 50:    # Strong wind (14+ m/s)
                score += 2
            elif wind > 30:    # Moderate wind (8+ m/s)
                score += 1
            
            # Visibility risk
            visibility = row.get('visibility', 20)
            if visibility < 0.5:    # Extremely poor visibility
                score += 3
            elif visibility < 2:    # Very poor visibility
                score += 2
            elif visibility < 5:    # Poor visibility
                score += 1
            
            # Humidity risk (extreme humidity affects comfort and health)
            humidity = row.get('humidity', 0.5)
            if isinstance(humidity, (int, float)):
                # Convert to percentage if it's between 0-1
                if humidity <= 1:
                    humidity = humidity * 100
                
                if humidity > 95 or humidity < 15:  # Extreme humidity
                    score += 2
                elif humidity > 85 or humidity < 25:  # High/low humidity
                    score += 1
            
            # Pressure risk (extreme pressure changes indicate storms)
            pressure = row.get('pressure', 1013)
            if pressure < 970 or pressure > 1040:  # Extreme pressure
                score += 2
            elif pressure < 990 or pressure > 1030:  # Abnormal pressure
                score += 1
            
            # Cloud cover risk
            cloud_cover = row.get('cloud_cover', 0.5)
            if isinstance(cloud_cover, (int, float)):
                if cloud_cover > 0.9:  # Very cloudy (potential for storms)
                    score += 1
            
            # Precipitation type risk
            precip_columns = [col for col in row.index if col.startswith('precip_') and col != 'precip_none']
            for precip_col in precip_columns:
                if row.get(precip_col, 0) == 1:
                    if 'snow' in precip_col or 'sleet' in precip_col:
                        score += 2
                    elif 'rain' in precip_col:
                        score += 1
            
            # Weather summary conditions
            if row.get('summary_fog', 0) == 1 or row.get('daily_summary_fog', 0) == 1:
                score += 2  # Fog is dangerous for travel
            
            if row.get('summary_wind', 0) == 1 or row.get('daily_summary_wind', 0) == 1:
                score += 1  # Windy conditions
            
            if row.get('summary_snow', 0) == 1 or row.get('daily_summary_snow', 0) == 1:
                score += 2  # Snow conditions
            
            # Heat index and wind chill considerations
            if 'heat_index' in row.index and row['heat_index'] > 40:
                score += 2
            elif 'heat_index' in row.index and row['heat_index'] > 35:
                score += 1
            
            if 'wind_chill' in row.index and row['wind_chill'] < -15:
                score += 2
            elif 'wind_chill' in row.index and row['wind_chill'] < -5:
                score += 1
            
            # Cap the score at 4 (Very High Risk)
            return min(score, 4)
        
        df['safety_score'] = df.apply(calculate_safety_score, axis=1)
        
        # Create categorical labels
        safety_labels = {
            0: 'Very Safe',
            1: 'Safe', 
            2: 'Moderate Risk',
            3: 'High Risk',
            4: 'Very High Risk'
        }
        
        df['safety_category'] = df['safety_score'].map(safety_labels)
        
        logger.info("Safety labels created successfully")
        logger.info(f"Label distribution:\n{df['safety_category'].value_counts()}")
        logger.info(f"Safety score statistics:\n{df['safety_score'].describe()}")
        
        return df

    def prepare_features(self, df):
        """
        Prepare features for machine learning
        """
        logger.info("Preparing features for training...")
        
        # Exclude non-feature columns
        exclude_cols = ['formatted_date', 'summary', 'daily_summary', 'precip_type', 
                       'safety_score', 'safety_category']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        
        logger.info(f"Selected features: {feature_cols}")
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        
        # Create additional interaction features
        if 'temperature' in X_imputed.columns and 'humidity' in X_imputed.columns:
            X_imputed['temp_humidity_interaction'] = X_imputed['temperature'] * X_imputed['humidity'] / 100
        
        if 'wind_speed' in X_imputed.columns and 'visibility' in X_imputed.columns:
            X_imputed['wind_visibility_interaction'] = X_imputed['wind_speed'] / (X_imputed['visibility'] + 1)
        
        if 'pressure' in X_imputed.columns:
            X_imputed['pressure_deviation'] = abs(X_imputed['pressure'] - 1013.25)  # Deviation from standard pressure
        
        self.feature_names = list(X_imputed.columns)
        
        y = df['safety_score']
        
        logger.info(f"Final features prepared. Shape: {X_imputed.shape}")
        logger.info(f"Target distribution: {y.value_counts().sort_index()}")
        
        return X_imputed, y

    def train_model(self, X, y):
        """
        Train the safety prediction model with comprehensive evaluation
        """
        logger.info("Starting model training...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Use XGBoost with GPU support. It will use GPU if built with CUDA and CUDA is available.
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [6, 10, 15],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.7, 1.0],
            'colsample_bytree': [0.7, 1.0]
        }

        # Detect Modal runtime environment and force CPU-only training inside Modal
        # Modal often sets environment variables like 'MODAL' or 'MODAL_RUN'. We'll check common ones.
        modal_env = any(k in os.environ for k in ("MODAL", "MODAL_RUN", "MODAL_APP"))
        if modal_env:
            logger.info("Modal environment detected via environment variables; forcing CPU-only training to avoid using Modal infra GPU")
            use_gpu_attempt = False
        else:
            use_gpu_attempt = True

        # Check for NVIDIA tooling presence to help debug GPU availability
        nvidia_smi_path = shutil.which('nvidia-smi')
        if nvidia_smi_path:
            logger.info(f"nvidia-smi found at: {nvidia_smi_path} (GPU driver likely present)")
        else:
            logger.info("nvidia-smi not found on PATH; GPU may not be available or drivers not in PATH")

        # Prepare XGBoost classifiers for both GPU and CPU modes
        xgb_gpu = XGBClassifier(
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            verbosity=1
        )

        xgb_cpu = XGBClassifier(
            tree_method='hist',
            predictor='cpu_predictor',
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            verbosity=1
        )

        # Decide training strategy
        if use_gpu_attempt:
            logger.info("Attempting hyperparameter tuning with XGBoost (GPU)...")
            grid_search = GridSearchCV(xgb_gpu, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
            try:
                grid_search.fit(X_train_scaled, y_train)
                self.model = grid_search.best_estimator_
                logger.info("GPU training succeeded")
                logger.info(f"Best parameters: {grid_search.best_params_}")
                logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            except Exception as e:
                # If GPU training fails, log the error and fallback to CPU
                logger.warning(f"GPU training failed with error: {e}")
                logger.info("Falling back to CPU XGBoost (hist/cpu_predictor)")
                grid_search_cpu = GridSearchCV(xgb_cpu, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
                grid_search_cpu.fit(X_train_scaled, y_train)
                self.model = grid_search_cpu.best_estimator_
                logger.info(f"Best CPU parameters: {grid_search_cpu.best_params_}")
                logger.info(f"Best CPU cross-validation score: {grid_search_cpu.best_score_:.4f}")
        else:
            # Modal detected: do not attempt GPU training
            logger.info("Skipping GPU attempt due to Modal environment. Running CPU XGBoost (hist/cpu_predictor)")
            grid_search_cpu = GridSearchCV(xgb_cpu, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
            grid_search_cpu.fit(X_train_scaled, y_train)
            self.model = grid_search_cpu.best_estimator_
            logger.info(f"Best CPU parameters: {grid_search_cpu.best_params_}")
            logger.info(f"Best CPU cross-validation score: {grid_search_cpu.best_score_:.4f}")

        # Log which tree_method and predictor are set on the final model
        try:
            tm = self.model.get_params().get('tree_method')
            pred = self.model.get_params().get('predictor')
            logger.info(f"Final XGBoost model tree_method={tm}, predictor={pred}")
        except Exception:
            logger.info("Final model parameters unavailable to inspect tree_method/predictor")

        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        # XGBoost predict_proba is available
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 15 Feature Importances:\n{feature_importance.head(15)}")
        
        # Class-wise performance
        safety_labels = {0: 'Very Safe', 1: 'Safe', 2: 'Moderate Risk', 3: 'High Risk', 4: 'Very High Risk'}
        
        for class_idx in range(len(safety_labels)):
            if class_idx in y_test.values:
                class_mask = (y_test == class_idx)
                if class_mask.sum() > 0:
                    class_accuracy = accuracy_score(y_test[class_mask], y_pred[class_mask])
                    logger.info(f"{safety_labels[class_idx]} accuracy: {class_accuracy:.4f}")
        
        return self.model, accuracy, feature_importance

    def predict_safety(self, weather_data):
        """
        Predict safety score for new weather data
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Ensure weather_data is a DataFrame
        if isinstance(weather_data, dict):
            weather_data = pd.DataFrame([weather_data])
        
        # Create the same features as during training
        if 'temperature' in weather_data.columns and 'humidity' in weather_data.columns:
            weather_data['temp_humidity_interaction'] = weather_data['temperature'] * weather_data['humidity'] / 100
        
        if 'wind_speed' in weather_data.columns and 'visibility' in weather_data.columns:
            weather_data['wind_visibility_interaction'] = weather_data['wind_speed'] / (weather_data['visibility'] + 1)
        
        if 'pressure' in weather_data.columns:
            weather_data['pressure_deviation'] = abs(weather_data['pressure'] - 1013.25)
        
        # Select only the features used during training
        available_features = [f for f in self.feature_names if f in weather_data.columns]
        missing_features = [f for f in self.feature_names if f not in weather_data.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Fill missing features with median values from training
            for feature in missing_features:
                weather_data[feature] = 0  # or use appropriate default values
        
        # Scale features
        weather_scaled = self.scaler.transform(weather_data[self.feature_names])
        
        # Predict
        prediction = self.model.predict(weather_scaled)
        probability = self.model.predict_proba(weather_scaled)
        
        safety_labels = {
            0: 'Very Safe',
            1: 'Safe', 
            2: 'Moderate Risk',
            3: 'High Risk',
            4: 'Very High Risk'
        }
        
        result = {
            'safety_score': int(prediction[0]),
            'safety_category': safety_labels[prediction[0]],
            'confidence': float(probability[0].max()),
            'probabilities': {
                safety_labels[i]: float(prob) 
                for i, prob in enumerate(probability[0])
            }
        }
        
        return result

    def save_model(self, filename='weather_safety_model_kaggle.pkl'):
        """
        Save the trained model and preprocessing objects
        """
        if self.model is None:
            raise ValueError("No model to save!")
        # XGBoost objects are picklable via joblib; save model and preprocessing artifacts
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'precip_encoder': self.precip_encoder
        }

        joblib.dump(model_data, filename)
        logger.info(f"Model and preprocessing objects saved to {filename}")

    def load_model(self, filename='weather_safety_model_kaggle.pkl'):
        """
        Load a trained model and preprocessing objects
        """
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.precip_encoder = model_data.get('precip_encoder', LabelEncoder())
            logger.info(f"Model loaded from {filename}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

def main():
    """
    Main execution function for training on Kaggle dataset
    """
    logger.info("Starting Weather Safety Model Training with Kaggle Dataset...")
    
    # Initialize the model
    weather_model = WeatherSafetyModel()
    
    # IMPORTANT: Replace with your actual dataset path
    dataset_path = "weatherHistory.csv"  # Update this path
    
    # Load the Kaggle dataset
    logger.info(f"Loading dataset from: {dataset_path}")
    df = weather_model.load_kaggle_dataset(dataset_path)
    
    if df is None:
        logger.error("Failed to load dataset. Please check the file path.")
        return
    
    # Preprocess the dataset
    df_processed = weather_model.preprocess_dataset(df)
    
    # Create safety labels
    df_labeled = weather_model.create_safety_labels(df_processed)
    
    # Prepare features
    X, y = weather_model.prepare_features(df_labeled)
    
    # Train model
    model, accuracy, feature_importance = weather_model.train_model(X, y)
    
    # Save model
    weather_model.save_model('weather_safety_model_kaggle.pkl')
    
    # Save feature importance for analysis
    feature_importance.to_csv('feature_importance.csv', index=False)
    logger.info("Feature importance saved to 'feature_importance.csv'")
    
    # Example prediction with Kaggle dataset format
    sample_weather = {
        'temperature': 25.0,
        'apparent_temperature': 28.0,
        'humidity': 0.65,  # Assuming this is in decimal format (0-1)
        'wind_speed': 15.0,
        'wind_bearing': 180.0,
        'visibility': 12.0,
        'cloud_cover': 0.3,
        'pressure': 1013.0,
        'month': 7,  # July
        'hour': 14,  # 2 PM
        'day_of_week': 2,  # Wednesday
        'season_encoded': 2,  # Summer
        'time_of_day_encoded': 2,  # Afternoon
        'month_sin': np.sin(2 * np.pi * 7 / 12),
        'month_cos': np.cos(2 * np.pi * 7 / 12),
        'hour_sin': np.sin(2 * np.pi * 14 / 24),
        'hour_cos': np.cos(2 * np.pi * 14 / 24),
        'summary_clear': 1,
        'summary_rain': 0,
        'daily_summary_clear': 1
    }
    
    try:
        prediction = weather_model.predict_safety(sample_weather)
        logger.info(f"Sample prediction: {prediction}")
    except Exception as e:
        logger.warning(f"Error in sample prediction: {e}")
    
    logger.info("Weather Safety Model training completed successfully!")
    logger.info(f"Final model accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()