import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# -----------------------------
# Load model and feature columns
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("knn_model.pkl")
    with open("feature_columns.json", "r") as f:
        feature_cols = json.load(f)
    return model, feature_cols

knn_model, FEATURE_COLS = load_model()

# Helper: build one-row DataFrame for prediction
def build_feature_row(
    age,
    monthly_spend,
    household_size,
    tenure_days,
    total_watch_events,
    total_watch_minutes,
    avg_watch_minutes,
    unique_movies_watched,
    avg_imdb_watched,
    share_high_imdb,
    gender,
    country,
    subscription_plan,
    primary_device,
):
    # start with all zeros
    data = {col: 0 for col in FEATURE_COLS}

    # numeric features
    def set_if_exists(col, value):
        if col in data:
            data[col] = value

    set_if_exists("age", age)
    set_if_exists("monthly_spend", monthly_spend)
    set_if_exists("household_size", household_size)
    set_if_exists("tenure_days", tenure_days)
    set_if_exists("total_watch_events", total_watch_events)
    set_if_exists("total_watch_minutes", total_watch_minutes)
    set_if_exists("avg_watch_minutes", avg_watch_minutes)
    set_if_exists("unique_movies_watched", unique_movies_watched)
    set_if_exists("avg_imdb_watched", avg_imdb_watched)
    set_if_exists("share_high_imdb", share_high_imdb)

    # gender one-hot
    gender_col = f"gender_{gender}"
    if gender_col in data:
        data[gender_col] = 1

    # country
    if country == "USA" and "country_USA" in data:
        data["country_USA"] = 1

    # subscription plan
    plan_col = f"subscription_plan_{subscription_plan}"
    if plan_col in data:
        data[plan_col] = 1

    # primary device
    device_col = f"primary_device_{primary_device}"
    if device_col in data:
        data[device_col] = 1

    df = pd.DataFrame([data])
    return df

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="KNN Churn Prediction",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("📺 Streaming Service – KNN Churn Prediction")

st.markdown(
    """
    This app uses a **K-Nearest Neighbors (KNN)** model to estimate the probability 
    that a user will **churn** (stop using the service).
    """
)

st.sidebar.header("User profile & usage")

# Numeric inputs
age = st.sidebar.slider("Age", 18, 80, 30)
household_size = st.sidebar.slider("Household size", 1, 8, 2)
monthly_spend = st.sidebar.slider("Monthly spend ($)", 0.0, 100.0, 20.0, 1.0)
tenure_days = st.sidebar.slider("Tenure (days)", 0, 365*3, 180)
total_watch_events = st.sidebar.slider("Total watch events", 0, 300, 50)
total_watch_minutes = st.sidebar.slider("Total watch minutes", 0, 100000, 5000, 10)
avg_watch_minutes = st.sidebar.slider("Average watch minutes per event", 0.0, 600.0, 100.0, 1.0)
unique_movies_watched = st.sidebar.slider("Unique movies watched", 0, 500, 20)
avg_imdb_watched = st.sidebar.slider("Average IMDb rating of watched movies", 0.0, 10.0, 7.0, 0.1)
share_high_imdb = st.sidebar.slider("Share of movies with IMDb ≥ 7.5 (%)", 0.0, 100.0, 40.0, 1.0)

# Categorical inputs
gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Other", "Prefer not to say"],
)
country = st.sidebar.selectbox("Country", ["USA", "Other"])
subscription_plan = st.sidebar.selectbox(
    "Subscription plan",
    ["Standard", "Premium", "Premium+"],
)
primary_device = st.sidebar.selectbox(
    "Primary device",
    ["Laptop", "Mobile", "Smart TV", "Tablet", "Gaming Console"],
)

if st.sidebar.button("Predict churn"):
    row = build_feature_row(
        age=age,
        monthly_spend=monthly_spend,
        household_size=household_size,
        tenure_days=tenure_days,
        total_watch_events=total_watch_events,
        total_watch_minutes=total_watch_minutes,
        avg_watch_minutes=avg_watch_minutes,
        unique_movies_watched=unique_movies_watched,
        avg_imdb_watched=avg_imdb_watched,
        share_high_imdb=share_high_imdb,
        gender=gender,
        country=country,
        subscription_plan=subscription_plan,
        primary_device=primary_device,
    )

    # Predict probability of churn (class 1)
    proba = knn_model.predict_proba(row)[0][1]
    churn_prob = float(proba) * 100

    st.subheader("Prediction")
    st.write(f"Estimated probability of churn: **{churn_prob:.1f}%**")

    if churn_prob >= 60:
        st.error("High risk of churn ⚠️")
    elif churn_prob >= 30:
        st.warning("Medium risk of churn 😐")
    else:
        st.success("Low risk of churn ✅")

    st.caption("Model: KNeighborsClassifier (scikit-learn)")
else:
    st.info("Set the inputs in the sidebar and click **Predict churn**.")
