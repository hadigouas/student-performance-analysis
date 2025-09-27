"""
Regression Models for Student Performance Prediction
This script trains and evaluates multiple regression models on the student performance dataset.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Constants
RANDOM_STATE = 42
MODELS_DIR = 'output/models'
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    """Load the preprocessed training and test data."""
    data_dir = 'output/features'
    
    X_train = np.load(f"{data_dir}/X_train.npy")
    X_test = np.load(f"{data_dir}/X_test.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")
    
    return X_train, X_test, y_train, y_test

def compute_metrics(y_true, y_pred, n_features):
    """Compute regression metrics."""
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate adjusted R-squared
    if n - n_features - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    else:
        adj_r2 = np.nan
    
    return {
        'r2': r2,
        'adj_r2': adj_r2,
        'rmse': rmse,
        'mae': mae
    }

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple regression models."""
    n_features = X_train.shape[1]
    results = []
    models = {}
    
    # Define models to train
    model_configs = [
        ('LinearRegression', LinearRegression()),
        ('DecisionTree', DecisionTreeRegressor(random_state=RANDOM_STATE)),
        ('RandomForest', RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE))
    ]
    
    for name, model in model_configs:
        print(f"\nTraining {name}...")
        start_time = time()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, n_features)
        metrics['training_time'] = time() - start_time
        
        # Store results
        results.append({
            'model': name,
            **metrics
        })
        
        # Store model
        models[name] = model
        
        print(f"  - R²: {metrics['r2']:.4f}")
        print(f"  - Adj. R²: {metrics['adj_r2']:.4f}")
        print(f"  - RMSE: {metrics['rmse']:.4f}")
        print(f"  - MAE: {metrics['mae']:.4f}")
        print(f"  - Training time: {metrics['training_time']:.2f}s")
    
    return pd.DataFrame(results), models

def save_best_model(results_df, models):
    """Save the best model based on adjusted R²."""
    # Find best model
    best_model_row = results_df.loc[results_df['adj_r2'].idxmax()]
    best_model_name = best_model_row['model']
    best_model = models[best_model_name]
    
    # Save the best model
    model_path = f"{MODELS_DIR}/best_regressor.joblib"
    joblib.dump(best_model, model_path)
    
    print(f"\nBest model: {best_model_name}")
    print(f"- Adj. R²: {best_model_row['adj_r2']:.4f}")
    print(f"- RMSE: {best_model_row['rmse']:.4f}")
    print(f"- Model saved to: {model_path}")
    
    return best_model_name

def main():
    """Main function to run the regression model training and evaluation."""
    print("Starting regression model training...")
    
    # Load data
    print("Loading preprocessed data...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Train and evaluate models
    print("Training regression models...")
    results_df, models = train_models(X_train, X_test, y_train, y_test)
    
    # Save results
    results_df.to_csv(f"{MODELS_DIR}/regression_results.csv", index=False)
    
    # Save best model
    best_model_name = save_best_model(results_df, models)
    
    # Save all models
    for name, model in models.items():
        joblib.dump(model, f"{MODELS_DIR}/{name.lower()}.joblib")
    
    print(f"\n✅ Regression training complete. Best model ({best_model_name}) saved to: {MODELS_DIR}")

if __name__ == "__main__":
    main()
