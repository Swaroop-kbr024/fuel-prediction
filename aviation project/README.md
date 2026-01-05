# ✈️ Aviation Fuel Consumption Prediction (ML Project)

This project predicts aircraft fuel consumption per flight using machine learning
and visualizes results through an interactive dashboard.

## Problem
Fuel cost is a major operational expense in aviation and directly impacts
carbon emissions.

## Solution
A regression-based ML model predicts fuel usage using:
- Flight distance
- Aircraft type
- Payload
- Cruise altitude
- Wind speed

CO₂ emissions are estimated using standard aviation conversion factors.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit

## How to Run
1. Train the model  
   `python train_model.py`

2. Launch the dashboard  
   `streamlit run app.py`

## Output
- Fuel consumption prediction (kg)
- CO₂ emission estimate (kg)

## Use Case
Aviation analytics, sustainability analysis, green-tech modeling
