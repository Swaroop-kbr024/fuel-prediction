import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "aviation_fuel_2025.csv"
MODEL_PATH = BASE_DIR / "fuel_model.pkl"
ENCODER_PATH = BASE_DIR / "aircraft_encoder.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Encode categorical column
encoder = LabelEncoder()
df["aircraft_type"] = encoder.fit_transform(df["aircraft_type"])

# Features & target
X = df.drop("fuel_consumed_kg", axis=1)
y = df["fuel_consumed_kg"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model & encoder
joblib.dump(model, MODEL_PATH)
joblib.dump(encoder, ENCODER_PATH)

print(f"✅ Model trained and saved successfully to {MODEL_PATH}")
