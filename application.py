import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set Streamlit page settings
st.set_page_config(page_title="AQI Dashboard", layout="wide")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Prediction", "About"])

# ----------------------- Load Data & Preprocess -----------------------

@st.cache_data
def load_and_clean_data():
    """Load and clean the dataset."""
    data = pd.read_csv("city_day.csv")

    data["Date"] = pd.to_datetime(data["Date"])
    
    numerical_columns = data.select_dtypes(include=['number']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())
    data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])
    data.drop_duplicates(inplace=True)

    return data

cleaned_data = load_and_clean_data()

# ----------------------- PAGE 1: HOME -----------------------

def home_page():
    """Displays AQI overview and categories."""
    st.title("🌍 Welcome to the AQI Dashboard")
    st.markdown("""
    ## 📌 What is Air Quality Index (AQI)?
    The **Air Quality Index (AQI)** measures air pollution levels and their impact on health.
    AQI is based on six pollutants:
    - PM2.5 & PM10
    - NO₂ (Nitrogen Dioxide)
    - SO₂ (Sulfur Dioxide)
    - O₃ (Ozone)
    - CO (Carbon Monoxide)
    """)

    st.markdown("## 📊 AQI Categories & Health Impacts")
    aqi_data = {
        "AQI Range": ["0-50", "51-100", "101-200", "201-300", "301-400", "401-500"],
        "Category": ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"],
    }
    st.table(aqi_data)

# ----------------------- PAGE 2: DATA ANALYSIS -----------------------

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    """Load and clean AQI dataset."""
    data = pd.read_csv("C:/Users/Ali/Desktop/AQI PROJECT/city_day.csv")

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

def data_analysis_page():
    """Displays AQI dataset and visualizations."""
    st.title("📊 Data Analysis & Visualization")

    # Show dataset preview
    st.subheader("🔍 Cleaned Dataset")
    st.write(cleaned_data.head(5))

    # Display summary statistics
    st.subheader("📋 Statistical Summary")
    st.write(cleaned_data.describe())

    # Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = cleaned_data[['PM2.5', 'PM10', 'NOx', 'SO2', 'CO', 'AQI']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Boxplot for Outliers
    st.subheader("📦 Boxplot for Pollutants")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=cleaned_data[['PM2.5', 'PM10', 'NOx', 'SO2', 'CO']], orient="h", palette="coolwarm")
    st.pyplot(fig)

    # AQI Category Bar Chart
    st.subheader("📌 AQI Category Distribution")
    group_by = cleaned_data.groupby("AQI_Bucket")["AQI"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=group_by.index, y=group_by.values, palette="coolwarm", ax=ax)
    plt.xlabel("AQI Category")
    plt.ylabel("Average AQI")
    plt.title("Average AQI per Category")
    st.pyplot(fig)

    st.sidebar.info("💡 Use this section to explore AQI data trends.")


# ----------------------- PAGE 3: AQI PREDICTION -----------------------

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@st.cache_data
def load_data():
    """Load and clean AQI dataset."""
    data = pd.read_csv("C:/Users/Ali/Desktop/AQI PROJECT/city_day.csv")

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

# ----------------- FIXED: Define train_model() before using it -----------------
def train_model():
    """Train a Random Forest model directly inside the app (without saving as .pkl)."""

    # Select Features & Target
    X = cleaned_data[["PM2.5", "PM10", "NO", "SO2", "O3", "CO"]]
    y = cleaned_data["AQI_Bucket"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model Trained! Accuracy: **{accuracy:.2f}**")

    return model

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

    # Train Model (Ensures train_model() is defined before calling)
    model = train_model()

    # Prediction Button
    if st.button("🔍 Predict AQI Category"):
        input_data = pd.DataFrame([[pm25, pm10, no, so2, o3, co]], columns=["PM2.5", "PM10", "NO", "SO2", "O3", "CO"])
        prediction = model.predict(input_data)[0]

        # Display prediction result
        st.success(f"🎯 Predicted AQI Category: **{prediction}**")


# ----------------------- PAGE 4: ABOUT -----------------------

import streamlit as st

def about_page():
    """Displays project details and team information."""
    st.title("ℹ️ About This Project")

    # Project Description
    st.markdown("""
    ## 🌎 Air Quality Index (AQI) Analysis & Prediction
    This project provides **real-time AQI analysis** and **AI-based AQI predictions** using machine learning.

    ### 🚀 Features:
    - **📊 Data Analysis**: Cleaned AQI dataset, statistical summaries, and visualizations.
    - **🤖 AI Predictions**: Machine learning model predicts AQI category based on pollutant levels.
    - **📈 Interactive Dashboard**: User-friendly interface for insights into air pollution trends.

    ---
    """)

    # Technologies Used
    st.subheader("🛠 Technologies Used")
    st.markdown("""
    - **Python**
    - **Streamlit**
    - **Pandas & NumPy**
    - **Seaborn & Matplotlib**
    - **Machine Learning (Random Forest)**
    """)

    st.markdown("---")

    # Team Members
    st.subheader("👨‍💻 Project Team")

    # Talha Khalid
    st.markdown("""
    ### 🏆 Talha Khalid
    - **🔗 LinkedIn:** [Talha Khalid](https://www.linkedin.com/in/talha-khalid-189092272)
    - **🐙 GitHub:** [talha142](https://github.com/talha142)
    - **📊 Kaggle:** [Talha Khalid](https://www.kaggle.com/talhachoudary/datasets)
    """)

    # Subhan Shahid
    st.markdown("""
    ### 🎯 Subhan Shahid
    - **🔗 LinkedIn:** [Subhan Shahid](https://linkedin.com/in/msubhanshahid)
    """)

    # Footer
    st.sidebar.info("💡 Explore other sections using the sidebar.")

# ----------------------- DISPLAY SELECTED PAGE -----------------------

if page == "Home":
    home_page()
elif page == "Data Analysis":
    data_analysis_page()
elif page == "Prediction":
    prediction_page()
elif page == "About":
    about_page()
