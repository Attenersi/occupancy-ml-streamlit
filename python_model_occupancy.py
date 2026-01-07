import sys
import subprocess
import importlib
import os
from pathlib import Path

# ==========================================
# 0. AUTO-INSTALL MISSING LIBRARIES
# ==========================================
def check_and_install_packages():
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'joblib': 'joblib' # Added joblib for saving models
    }

    for import_name, package_name in required_packages.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"Installing {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

check_and_install_packages()

import pandas as pd
import numpy as np
import warnings
import joblib  # <--- NEW IMPORT
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "training_datasets"
OUTPUT_DIR = SCRIPT_DIR
COLUMN_NAMES = ['Id', 'Date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'Occupancy']

models_config = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision_Tree": DecisionTreeClassifier(max_depth=10, random_state=42)
}

# ==========================================
# 2. SHARED FEATURE ENGINEERING
# ==========================================
# NOTE: This logic is CRITICAL. It must match the App exactly.
def smart_feature_engineering(df, is_training=True):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Time
    df['Hour'] = df['Date'].dt.hour
    df['Is_Weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour']/24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour']/24)
    
    # Rolling & Deltas
    window_size = 4
    df['CO2_Smooth'] = df['CO2'].rolling(window=window_size).mean()
    df['Temp_Smooth'] = df['Temperature'].rolling(window=window_size).mean()
    df['CO2_Delta'] = df['CO2'] - df['CO2'].shift(window_size)
    df['Temp_Delta'] = df['Temperature'] - df['Temperature'].shift(window_size)
    df['CO2_Lag1'] = df['CO2'].shift(window_size)
    
    # Handling NaNs
    if is_training:
        df = df.dropna()
    else:
        df['CO2_Delta'] = df['CO2_Delta'].fillna(0)
        df['Temp_Delta'] = df['Temp_Delta'].fillna(0)
        df['CO2_Smooth'] = df['CO2_Smooth'].fillna(df['CO2'])
        df['Temp_Smooth'] = df['Temp_Smooth'].fillna(df['Temperature'])
        df['CO2_Lag1'] = df['CO2_Lag1'].fillna(df['CO2'])
        
    return df

def load_data():
    try:
        t1 = pd.read_csv(DATA_DIR / 'datatraining.txt', names=COLUMN_NAMES, header=None, skiprows=1)
        t2 = pd.read_csv(DATA_DIR / 'datatraining2.txt', names=COLUMN_NAMES, header=None, skiprows=1)
        return pd.concat([t1, t2], ignore_index=True)
    except Exception as e:
        print(f"Data load error: {e}")
        return None

# ==========================================
# 3. MAIN TRAINING & SAVING
# ==========================================
def main():
    print("--- 1. Loading & Training Models ---")
    df_train = load_data()
    if df_train is None: return

    df_train = smart_feature_engineering(df_train, is_training=True)

    feature_cols = [
        'Temperature', 'Humidity', 'CO2', 'HumidityRatio', 
        'Is_Weekend', 'Hour_Sin', 'Hour_Cos',
        'CO2_Smooth', 'Temp_Smooth', 'CO2_Delta', 'Temp_Delta', 'CO2_Lag1'
    ]
    target_col = 'Occupancy'

    X = df_train[feature_cols]
    y = df_train[target_col]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[('scaler', StandardScaler(), feature_cols)]
    )

    # Dictionary to hold trained pipelines
    trained_models = {}

    for name, model in models_config.items():
        print(f"Training {name}...")
        # Create Pipeline
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        # Fit on FULL training data
        pipeline.fit(X, y)
        # Store in dictionary
        trained_models[name] = pipeline

    # --- 4. SAVING THE BUNDLE ---
    print("\n--- 2. Saving Model Bundle ---")
    
    bundle = {
        "models": trained_models,       # All 4 trained pipelines
        "features": feature_cols,       # The list of columns required
        "timestamp": pd.Timestamp.now() # Metadata
    }
    
    save_path = OUTPUT_DIR / "occupancy_model_bundle.pkl"
    joblib.dump(bundle, save_path)
    
    print(f"SUCCESS: Models saved to {save_path}")
    print("You can now run the Streamlit App.")

if __name__ == "__main__":
    main()