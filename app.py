import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------- LOAD FILES ----------------
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")   # IMPORTANT

# ---------------- TITLE ----------------
st.set_page_config(page_title="Sales_Revenue Prediction", layout="centered")
st.title("Sales Revenue Prediction System")
st.write("Enter marketing details:")


# ---------------- NUMERICAL INPUTS ----------------
ad_spend = st.number_input("Ad Spend", min_value=0.5, max_value=1250.0, value=10.0, format="%.4f")

price = st.number_input("Price", min_value=0.2, max_value=150.0, value=1.0, format="%.4f")

discount_rate = st.number_input("Discount Rate", min_value=0.0, max_value=0.6, value=0.1, format="%.4f")

market_reach = st.number_input("Market Reach", min_value=10.0, max_value=1200.0, value=100.0)

impressions = st.number_input("Impressions", min_value=100, max_value=36000, value=5000)

ctr = st.number_input("Click Through Rate", min_value=0.0, max_value=0.25, value=0.05, format="%.4f")

competition_index = st.number_input("Competition Index", min_value=0.0, max_value=10.0, value=5.0, format="%.4f")

seasonality_index = st.number_input("Seasonality Index", min_value=-1.5, max_value=1.5, value=0.0, format="%.4f")

campaign_days = st.number_input("Campaign Duration (days)", min_value=7, max_value=90, value=30)

clv = st.number_input("Customer Lifetime Value", min_value=20.0, max_value=200000.0, value=1000.0, format="%.4f")

# ---------------- CATEGORICAL INPUTS ----------------
region = st.selectbox("Region", ["Central", "East", "North", "South", "West"])

channel = st.selectbox("Channel", ["Affiliate", "Email", "Influencer", "Search", "Social Media", "Tv"])

product = st.selectbox("Product Category", ["General", "Kitchen", "Lighting", "Seasonal", "Stationery", "Storage"])

segment = st.selectbox("Customer Segment", ["Budget", "Premium", "Standard"])

# ---------------- CREATE INPUT ----------------
input_dict = {}

# Fill all columns with 0
for col in columns:
    input_dict[col] = 0

# Add numerical values
input_dict['ad_spend'] = ad_spend
input_dict['price'] = price
input_dict['discount_rate'] = discount_rate
input_dict['market_reach'] = market_reach
input_dict['impressions'] = impressions
input_dict['click_through_rate'] = ctr
input_dict['competition_index'] = competition_index
input_dict['seasonality_index'] = seasonality_index
input_dict['campaign_duration_days'] = campaign_days
input_dict['customer_lifetime_value'] = clv

# Set categorical = 1
input_dict[f"region_{region}"] = 1
input_dict[f"channel_{channel}"] = 1
input_dict[f"product_category_{product}"] = 1
input_dict[f"customer_segment_{segment}"] = 1

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Ensure correct column order
input_df = input_df[columns]

# ---------------- SCALE ----------------
input_scaled = scaler.transform(input_df)

# ---------------- PREDICT ----------------
if st.button("Predict Sales Revenue"):
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Sales Revenue: {prediction[0]:,.2f}")