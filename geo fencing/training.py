# FILE: training.py
# --- UPDATED FOR REGRESSION ---

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor  # CHANGED: Using Regressor
from sklearn.metrics import mean_squared_error, r2_score # CHANGED: Using regression metrics
import numpy as np
import joblib

# Load the data prepared by features.py
df = pd.read_csv("geofence_features.csv")

# --- DEFINE FEATURES AND NEW TARGET ---
# We are training the model on the hardest, most realistic problem:
# predicting the score from only the raw coordinates.
X = df[["latitude", "longitude"]]
y = df["safety_score"]  # CHANGED: Our target is now the continuous score

# --- TRAIN/TEST SPLIT ---
# No 'stratify' is needed for regression targets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- HYPERPARAMETER TUNING FOR REGRESSION MODEL ---

# 1. Define the base regression model
model_base = RandomForestRegressor(random_state=42)

# 2. Define the grid of parameters to search
# We can use a slightly smaller grid for faster tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [20, None],          # Let trees be deep to capture complex boundaries
    'min_samples_split': [2, 5]       # Standard values
}

# 3. Set up the Grid Search object
# For regressors, it automatically uses R-squared for scoring, which is perfect.
grid_search = GridSearchCV(estimator=model_base, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2)

# 4. Fit the Grid Search to the training data
print("Starting hyperparameter tuning for REGRESSION model...")
grid_search.fit(X_train, y_train)

# 5. Get the best performing model from the search
best_model = grid_search.best_estimator_
print(f"\nBest parameters found: {grid_search.best_params_}")


# --- EVALUATE THE BEST MODEL ---
print("\nEvaluating the best REGRESSION model on the test set:")
y_pred = best_model.predict(X_test)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print("\nInterpretation:")
print(f"-> RMSE means our model's score predictions are, on average, off by about +/- {rmse:.2f} points.")
print(f"-> R² of 1.0 is a perfect prediction. A score above 0.9 is excellent.")


# --- SAVE THE FINAL MODEL ---
# The LabelEncoder is no longer needed.
joblib.dump(best_model, "geofence_safety_model.pkl")
print("\nBest REGRESSION model saved as geofence_safety_model.pkl")