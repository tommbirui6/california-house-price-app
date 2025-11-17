import streamlit as st
import pandas as pd
import joblib

# Load trained model
@st.cache_data
def load_model():
    return joblib.load("california_knn_pipeline.pkl")

model = load_model()

# App title
st.set_page_config(page_title="California House Price Predictor", layout="wide")
st.title("🏠 California House Price Predictor")
st.markdown("Adjust the sliders to enter house features and predict the median house price.")

# Sidebar inputs
st.sidebar.header("Enter House Features")
longitude = st.sidebar.slider("Longitude", -124.35, -114.31, -120.0)
latitude = st.sidebar.slider("Latitude", 32.54, 42.01, 34.0)
housing_median_age = st.sidebar.slider("Housing Median Age", 1, 52, 20)
total_rooms = st.sidebar.slider("Total Rooms", 2, 10000, 1000)
total_bedrooms = st.sidebar.slider("Total Bedrooms", 1, 5000, 500)
population = st.sidebar.slider("Population", 3, 10000, 1500)
households = st.sidebar.slider("Households", 1, 5000, 500)
median_income = st.sidebar.slider("Median Income (10k USD)", 0.0, 20.0, 5.0)

# Input dataframe
input_data = pd.DataFrame({
    'longitude': [longitude],
    'latitude': [latitude],
    'housing_median_age': [housing_median_age],
    'total_rooms': [total_rooms],
    'total_bedrooms': [total_bedrooms],
    'population': [population],
    'households': [households],
    'median_income': [median_income]
})

# Prediction
if st.button("Predict House Price"):
    try:
        prediction = model.predict(input_data)
        st.success(f"💰 Estimated House Price: ${prediction[0]*100000:,.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
