import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Resolve model paths relative to this script so Streamlit finds them reliably
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fuel_model.pkl"
ENCODER_PATH = BASE_DIR / "aircraft_encoder.pkl"

# Load model and encoder with clear errors if missing
if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not ENCODER_PATH.exists():
    st.error(f"Encoder file not found: {ENCODER_PATH}")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed loading model: {e}")
    st.stop()

try:
    encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    st.error(f"Failed loading encoder: {e}")
    st.stop()

st.set_page_config(page_title="Aviation Fuel ML", layout="centered")
st.title("✈️ Aircraft Fuel Consumption Prediction")

st.markdown("Predict **fuel usage and CO₂ emissions** for a flight")

# Inputs
distance = st.slider("Flight Distance (km)", 500, 3000, 1200)
aircraft = st.selectbox("Aircraft Type", ["A320", "B737", "A321", "B787"])
payload = st.slider("Payload (kg)", 10000, 30000, 16000)
altitude = st.slider("Cruise Altitude (ft)", 30000, 42000, 35000)
wind = st.slider("Wind Speed (km/h)", 0, 100, 30)

# Prediction
if st.button("Predict Fuel Consumption"):
    aircraft_encoded = encoder.transform([aircraft])[0]

    input_df = pd.DataFrame([[
        distance,
        aircraft_encoded,
        payload,
        altitude,
        wind
    ]], columns=[
        "distance_km",
        "aircraft_type",
        "payload_kg",
        "cruise_altitude_ft",
        "wind_speed_kmh"
    ])

    fuel = model.predict(input_df)[0]
    co2 = fuel * 3.16

    st.success(f"🛢️ Fuel Consumed: **{fuel:.2f} kg**")
    st.info(f"🌱 Estimated CO₂ Emission: **{co2:.2f} kg**")
