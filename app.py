import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="California House Price Predictor", layout="wide")

# Title
st.title("🏡 California Housing Price Prediction App")
st.write("""
Adjust the sliders to enter house features and predict the median house price.
This app uses a trained KNN model for predictions and displays interactive metrics and data insights.
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

# Make prediction
if st.button("🔮 Predict House Price"):
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

# Interactive model performance metrics
st.header("2️⃣ Model Performance Metrics (Example)")

# You can replace these with your real metrics if you have them
rmse = st.slider("RMSE (Root Mean Squared Error)", 10000, 100000, 52000)
r2 = st.slider("R² Score", 0.0, 1.0, 0.79)

st.metric("RMSE", f"${rmse:,.0f}")
st.metric("R² Score", f"{r2:.2f}")

# Optional data visualizations / insights
st.header("3️⃣ Data Insights")

st.write("Correlation heatmap of housing features (example).")

# Example data for visualization (replace with your real dataset if available)
data = pd.DataFrame({
    "MedInc": [median_income, 5, 7, 3, 8, 4],
    "HouseAge": [housing_median_age, 20, 30, 15, 40, 25],
    "AveRooms": [total_rooms, 3000, 4000, 2000, 5000, 3500],
    "AveBedrms": [total_bedrooms, 500, 700, 300, 800, 600],
    "Population": [population, 1500, 2000, 1000, 2500, 1800],
    "AveOccup": [households, 500, 600, 400, 700, 550]
})

# Plot correlation heatmap
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Scatterplot example
st.write("Scatterplot: Median Income vs. Population")
fig2, ax2 = plt.subplots()
sns.scatterplot(x="MedInc", y="Population", data=data, ax=ax2)
st.pyplot(fig2)
