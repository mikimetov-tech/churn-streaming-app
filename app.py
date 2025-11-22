import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# ---------------------------
# Page config & basic styling
# ---------------------------
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Optional: if you add a logo.png to your repo, uncomment this:
# st.image("logo.png", width=150)

st.title("🎬 User Churn Prediction Dashboard")
st.markdown(
    """
This app uses a **Random Forest** model trained on:
- User demographics & subscription info  
- Watch history behaviour  
- Movie-level attributes (e.g., IMDb ratings)  

to estimate the **probability that a user will churn**.
"""
)

# ---------------------------
# Load model & feature columns
# ---------------------------
@st.cache_resource
def load_model_and_features():
    rf_model = joblib.load("rf_model.pkl")
    with open("feature_columns.json", "r") as f:
        feature_cols = json.load(f)
    return rf_model, feature_cols

rf_model, FEATURE_COLS = load_model_and_features()

# ---------------------------
# Helper: build input row like training X
# ---------------------------
def build_feature_row(
    age,
    household_size,
    monthly_spend,
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
    primary_device
):
    # start with all zeros
    data = {col: 0.0 for col in FEATURE_COLS}

    # numeric features
    num_map = {
        "age": age,
        "household_size": household_size,
        "monthly_spend": monthly_spend,
        "tenure_days": tenure_days,
        "total_watch_events": total_watch_events,
        "total_watch_minutes": total_watch_minutes,
        "avg_watch_minutes": avg_watch_minutes,
        "unique_movies_watched": unique_movies_watched,
        "avg_imdb_watched": avg_imdb_watched,
        "share_high_imdb": share_high_imdb,
    }
    for col, val in num_map.items():
        if col in data:
            data[col] = val

    # gender (baseline: Female)
    if gender == "Male" and "gender_Male" in data:
        data["gender_Male"] = 1
    elif gender == "Other" and "gender_Other" in data:
        data["gender_Other"] = 1
    elif gender == "Prefer not to say" and "gender_Prefer not to say" in data:
        data["gender_Prefer not to say"] = 1

    # country (baseline: Other)
    if country == "USA" and "country_USA" in data:
        data["country_USA"] = 1

    # subscription plan (baseline: Basic)
    if subscription_plan == "Standard" and "subscription_plan_Standard" in data:
        data["subscription_plan_Standard"] = 1
    elif subscription_plan == "Premium" and "subscription_plan_Premium" in data:
        data["subscription_plan_Premium"] = 1
    elif subscription_plan == "Premium+" and "subscription_plan_Premium+" in data:
        data["subscription_plan_Premium+"] = 1

    # primary device (baseline: Desktop/Other)
    dev_map = {
        "Mobile": "primary_device_Mobile",
        "Laptop": "primary_device_Laptop",
        "Tablet": "primary_device_Tablet",
        "Smart TV": "primary_device_Smart TV",
        "Gaming Console": "primary_device_Gaming Console",
    }
    if primary_device in dev_map and dev_map[primary_device] in data:
        data[dev_map[primary_device]] = 1

    # return DataFrame in correct column order
    return pd.DataFrame([data])[FEATURE_COLS]


# ---------------------------
# Sidebar: navigation + inputs
# ---------------------------
mode = st.sidebar.radio("Select view:", ["🔮 Prediction", "📊 Model Insights"])

st.sidebar.markdown("### User Features")

age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30)
household_size = st.sidebar.number_input("Household Size", min_value=1, max_value=10, value=3)
monthly_spend = st.sidebar.number_input("Monthly Spend ($)", min_value=0.0, value=15.0, step=1.0)
tenure_days = st.sidebar.number_input("Tenure (days since subscription start)", min_value=0, value=180)

gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other", "Prefer not to say"])
country = st.sidebar.selectbox("Country", ["Other", "USA"])
subscription_plan = st.sidebar.selectbox("Subscription Plan", ["Basic", "Standard", "Premium", "Premium+"])
primary_device = st.sidebar.selectbox(
    "Primary Device",
    ["Desktop/Other", "Mobile", "Laptop", "Tablet", "Smart TV", "Gaming Console"]
)

st.sidebar.markdown("### Engagement & Content Behaviour")
total_watch_events = st.sidebar.number_input("Total Watch Sessions", min_value=0, value=20)
total_watch_minutes = st.sidebar.number_input("Total Watch Minutes", min_value=0.0, value=600.0, step=10.0)
avg_watch_minutes = st.sidebar.number_input("Average Minutes per Session", min_value=0.0, value=30.0, step=1.0)
unique_movies_watched = st.sidebar.number_input("Unique Movies Watched", min_value=0, value=10)
avg_imdb_watched = st.sidebar.number_input(
    "Average IMDb Rating of Watched Movies", min_value=0.0, max_value=10.0, value=7.0, step=0.1
)
share_high_imdb = st.sidebar.slider(
    "Share of titles with IMDb ≥ 7.5", min_value=0.0, max_value=1.0, value=0.5, step=0.05
)

# ---------------------------
# MAIN VIEW: Prediction
# ---------------------------
if mode == "🔮 Prediction":
    st.subheader("🔮 Churn Probability Estimation")

    if st.button("Predict Churn Probability"):
        row = build_feature_row(
            age, household_size, monthly_spend, tenure_days,
            total_watch_events, total_watch_minutes,
            avg_watch_minutes, unique_movies_watched,
            avg_imdb_watched, share_high_imdb,
            gender, country, subscription_plan, primary_device
        )

        proba = rf_model.predict_proba(row)[0][1]

        st.write(f"### Estimated Churn Probability: **{proba:.3f}**")

        if proba >= 0.5:
            st.error("⚠️ High Churn Risk – consider a retention campaign.")
        elif proba >= 0.3:
            st.warning("⚠️ Medium Churn Risk – monitor and engage this user.")
        else:
            st.success("✅ Low Churn Risk – user is likely to stay.")

        with st.expander("See model input row (debug / explanation)"):
            st.dataframe(row)

# ---------------------------
# MAIN VIEW: Model Insights
# ---------------------------
else:
    st.subheader("📊 Model Insights: Feature Importances")

    importances = rf_model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    top_n = st.slider("Number of top features to display", 5, min(25, len(FEATURE_COLS)), 15)

    top_feat = feat_imp.head(top_n).iloc[::-1]  # reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_feat["feature"], top_feat["importance"])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top Feature Importances – Random Forest")
    plt.tight_layout()

    st.pyplot(fig)

    st.markdown(
        """
**How to read this chart:**

- Features at the top contribute the most to the model's decision.  
- For example, higher values of **engagement** (watch minutes, sessions, unique titles)  
  are usually associated with **lower churn risk**.  
- Shorter **tenure** and low **usage** can indicate higher churn risk.
"""
    )

    with st.expander("Full feature importance table"):
        st.dataframe(feat_imp.reset_index(drop=True))
