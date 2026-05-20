import streamlit as st
import joblib

# Load model files
model = joblib.load("scheduler_model.pkl")
country_encoder = joblib.load("country_encoder.pkl")
content_encoder = joblib.load("content_encoder.pkl")

st.title("📊 AI Marketing Scheduler")

country = st.text_input("Enter Country")
platform = st.text_input("Enter Platform (Instagram/TikTok/YouTube)")
content = st.text_input("Enter Content Type")

if st.button("Predict Best Time"):
    # Example prediction logic (depends on your model)
    st.success("Best time to post: 6:00 PM 🚀")
