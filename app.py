
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
from datetime import date
import os

st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🚴",
    layout="centered"
)

st.title("🚴 Food Delivery Time Prediction")
st.write("Predict the estimated delivery time of a food order.")

# Load saved models
@st.cache_resource
def load_models():
    preprocessor = joblib.load("preprocessor.pkl")
    xgb_model = joblib.load("xgboost_model.pkl")
    lgbm_model = joblib.load("lightgbm_model.pkl")
    meta_model = joblib.load("meta_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    cat_model = CatBoostRegressor()
    cat_model.load_model("catboost_model.cbm")

    return (
        preprocessor, xgb_model, lgbm_model,
        cat_model, meta_model, feature_columns
    )

try:
    (
        preprocessor, xgb_model, lgbm_model,
        cat_model, meta_model, feature_columns
    ) = load_models()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# Get categories learned during training
cat_columns = [
    "Day_of_Week", "Weather", "Pickup_Zone",
    "Dropoff_Zone", "Vehicle_Type", "Cuisine_Type",
    "Restaurant_Load", "Delivery_Distance_Category",
    "Traffic_Level", "Delivery_Priority"
]

category_values = dict(zip(
    cat_columns,
    preprocessor.named_transformers_["cat"]
    .named_steps["onehot"].categories_
))

with st.form("delivery_form"):
    st.subheader("Order Details")

    order_date = st.date_input("Order Date", value=date.today())
    order_hour = st.slider("Order Hour", 0, 23, 19)

    day_name = order_date.strftime("%A")
    day_options = list(category_values["Day_of_Week"])
    if day_name not in day_options:
        day_name = day_options[0]

    weather = st.selectbox("Weather", list(category_values["Weather"]))
    pickup = st.selectbox("Pickup Zone", list(category_values["Pickup_Zone"]))
    dropoff = st.selectbox("Dropoff Zone", list(category_values["Dropoff_Zone"]))
    vehicle = st.selectbox("Vehicle Type", list(category_values["Vehicle_Type"]))
    cuisine = st.selectbox("Cuisine Type", list(category_values["Cuisine_Type"]))
    restaurant_load = st.selectbox(
        "Restaurant Load", list(category_values["Restaurant_Load"])
    )
    distance_category = st.selectbox(
        "Delivery Distance Category",
        list(category_values["Delivery_Distance_Category"])
    )
    traffic = st.selectbox(
        "Traffic Level", list(category_values["Traffic_Level"])
    )
    priority = st.selectbox(
        "Delivery Priority", list(category_values["Delivery_Priority"])
    )

    st.subheader("Delivery Details")

    festival = st.checkbox("Is it a festival day?")
    rider_experience = st.number_input(
        "Rider Experience (years)", 0.0, 40.0, 3.0, 0.5
    )
    rider_rating = st.slider("Rider Rating", 1.0, 5.0, 4.5, 0.1)
    restaurant_rating = st.slider(
        "Restaurant Rating", 1.0, 5.0, 4.0, 0.1
    )
    order_items = st.number_input("Number of Items", 1, 30, 3)
    preparation_time = st.number_input(
        "Preparation Time (minutes)", 1, 120, 15
    )
    road_distance = st.number_input(
        "Road Distance (km)", 0.1, 100.0, 5.0, 0.1
    )
    signals = st.number_input("Number of Traffic Signals", 0, 100, 5)
    average_speed = st.number_input(
        "Average Speed (km/h)", 1.0, 120.0, 25.0, 1.0
    )

    submitted = st.form_submit_button("Predict Delivery Time")

if submitted:
    hour = order_hour
    month = order_date.month

    # Engineer features in the same format as training
    input_data = {
        "Order_Hour": hour,
        "Day_of_Week": day_name,
        "Is_Weekend": int(order_date.weekday() >= 5),
        "Is_Festival": int(festival),
        "Weather": weather,
        "Pickup_Zone": pickup,
        "Dropoff_Zone": dropoff,
        "Vehicle_Type": vehicle,
        "Rider_Experience_Years": rider_experience,
        "Rider_Rating": rider_rating,
        "Restaurant_Rating": restaurant_rating,
        "Cuisine_Type": cuisine,
        "Order_Items": order_items,
        "Restaurant_Load": restaurant_load,
        "Preparation_Time_Min": preparation_time,
        "Road_Distance_km": road_distance,
        "Delivery_Distance_Category": distance_category,
        "Traffic_Level": traffic,
        "Number_of_Signals": signals,
        "Average_Speed_kmph": average_speed,
        "Delivery_Priority": priority,
        "Order_Year": order_date.year,
        "Order_Month": month,
        "Order_Day": order_date.day,
        "Is_Peak_Hour": int(
            hour in [12, 13, 14, 18, 19, 20, 21]
        ),
        "Hour_Sin": np.sin(2 * np.pi * hour / 24),
        "Hour_Cos": np.cos(2 * np.pi * hour / 24),
        "Month_Sin": np.sin(2 * np.pi * month / 12),
        "Month_Cos": np.cos(2 * np.pi * month / 12),
        "Signals_per_km": signals / road_distance,
        "Preparation_Time_per_Item": preparation_time / order_items
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_columns]

    try:
        processed = preprocessor.transform(input_df)

        xgb_pred = xgb_model.predict(processed)
        lgbm_pred = lgbm_model.predict(processed)
        cat_pred = cat_model.predict(input_df)

        base_predictions = np.column_stack([
            xgb_pred, lgbm_pred, cat_pred
        ])

        predicted_minutes = float(
            meta_model.predict(base_predictions)[0]
        )

        st.success("Prediction completed!")
        st.metric(
            "Estimated Delivery Time",
            f"{max(0, predicted_minutes):.1f} minutes"
        )
        st.caption(
            "This is an ML-based estimate; actual delivery time may vary."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
