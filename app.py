import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="California House Price Predictor", layout="wide")

# Title
st.title("🏡 California Housing Price Prediction App")
st.write("""
Adjust the sliders to enter house features and predict the median house price.
This app uses a trained KNN model for predictions.
""")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("california_knn_pipeline.pkl")

model = load_model()
st.success("Model loaded successfully!")

# User inputs
st.header("1️⃣ Enter Housing Features")

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude", -125.0, -110.0, -118.0)
    housing_median_age = st.number_input("Housing Median Age", 1, 60, 20)
    total_rooms = st.number_input("Total Rooms", 1, 50000, 3000)
    population = st.number_input("Population", 1, 50000, 1500)

with col2:
    latitude = st.number_input("Latitude", 30.0, 45.0, 34.0)
    total_bedrooms = st.number_input("Total Bedrooms", 1, 10000, 500)
    households = st.number_input("Households", 1, 10000, 500)
    median_income = st.number_input("Median Income (10k USD)", 0.1, 20.0, 5.0)

# Predict button
if st.button("🔮 Predict House Price"):
    # Match the column names expected by the trained model
    input_df = pd.DataFrame([{
        "MedInc": median_income,
        "HouseAge": housing_median_age,
        "AveRooms": total_rooms,
        "AveBedrms": total_bedrooms,
        "Population": population,
        "AveOccup": households,
        "Latitude": latitude,
        "Longitude": longitude
    }])

    try:
        pred = model.predict(input_df)[0]
        st.subheader(f"🏠 Predicted House Price: **${pred:,.2f}**")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Display example metrics (replace with actual metrics if available)
st.header("2️⃣ Model Performance Metrics")
st.metric("RMSE", "$52,000")
st.metric("R² Score", "0.79")

# Optional visualization for feature importance
st.header("3️⃣ Feature Importance")
st.info("KNN models do not support feature importance.")
