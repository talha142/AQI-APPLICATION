import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set Streamlit page settings
st.set_page_config(page_title="AQI Dashboard", layout="wide")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Prediction", "About"])

# ----------------------- Load Data -----------------------

@st.cache_data
def load_data():
    """Load and clean AQI dataset."""
    data = pd.read_csv("city_day.csv")

    # Drop 'City' and 'Date' columns
    data.drop(columns=["City", "Date"], inplace=True, errors="ignore")

    # Fill missing values
    numerical_columns = data.select_dtypes(include=['number']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())
    data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

    # Remove duplicates
    data.drop_duplicates(inplace=True)

    return data

cleaned_data = load_data()

# ----------------- Train Model with Caching -----------------
@st.cache_resource
def train_model():
    """Train a Random Forest model and cache it."""
    
    # Select Features & Target
    features = ["PM2.5", "PM10", "NO", "SO2", "O3", "CO"]
    
    # Ensure 'AQI_Bucket' exists
    if "AQI_Bucket" not in cleaned_data.columns:
        st.error("Error: 'AQI_Bucket' column is missing in the dataset!")
        return None
    
    X = cleaned_data[features]
    y = cleaned_data["AQI_Bucket"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.success(f"✅ Model Trained! Accuracy: **{accuracy:.2f}**")

    return model

# Train the model only once (cached)
model = train_model()

# ----------------- Prediction Page -----------------
def prediction_page():
    """Allows users to enter pollutant levels and get an AQI prediction."""
    st.title("📈 AQI Prediction")

    st.markdown("### 🔢 Enter Pollutant Levels to Predict AQI Category")

    # User input form layout
    col1, col2 = st.columns(2)
    
    with col1:
        pm25 = st.number_input("PM2.5 Level (µg/m³)", 0.0, 500.0, 50.0, step=1.0)
        pm10 = st.number_input("PM10 Level (µg/m³)", 0.0, 500.0, 80.0, step=1.0)
        no = st.number_input("NO Level (µg/m³)", 0.0, 500.0, 30.0, step=1.0)

    with col2:
        so2 = st.number_input("SO2 Level (µg/m³)", 0.0, 500.0, 10.0, step=1.0)
        o3 = st.number_input("O3 Level (µg/m³)", 0.0, 500.0, 20.0, step=1.0)
        co = st.number_input("CO Level (ppm)", 0.0, 500.0, 1.0, step=0.1)

    # Prediction Button
    if st.button("🔍 Predict AQI Category"):
        if model is None:
            st.error("Model is not available. Please check data processing.")
        else:
            input_data = pd.DataFrame([[pm25, pm10, no, so2, o3, co]], 
                                      columns=["PM2.5", "PM10", "NO", "SO2", "O3", "CO"])
            prediction = model.predict(input_data)[0]

            # Display prediction result
            st.success(f"🎯 Predicted AQI Category: **{prediction}**")

# ----------------- Page Navigation -----------------

if page == "Home":
    st.title("🌍 Welcome to the AQI Dashboard")
    st.write("Explore air quality data and make predictions.")

elif page == "Data Analysis":
    st.title("📊 Data Analysis & Visualization")
    st.write(cleaned_data.head())

elif page == "Prediction":
    prediction_page()

elif page == "About":
    st.title("ℹ️ About This Project")
    st.write("A machine learning-based AQI prediction dashboard using Streamlit.")
