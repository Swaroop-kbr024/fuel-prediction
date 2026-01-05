import joblib
import pandas as pd

model = joblib.load("fuel_model.pkl")
encoder = joblib.load("aircraft_encoder.pkl")

# Build a sample input matching app.py
aircraft = "A320"
aircraft_encoded = encoder.transform([aircraft])[0]

input_df = pd.DataFrame([[
    1200,
    aircraft_encoded,
    16000,
    35000,
    30
]], columns=[
    "distance_km",
    "aircraft_type",
    "payload_kg",
    "cruise_altitude_ft",
    "wind_speed_kmh"
])

print("Input:", input_df.to_dict(orient='records')[0])
pred = model.predict(input_df)[0]
print(f"Predicted fuel: {pred:.2f} kg")
