import streamlit as st
import joblib
import numpy as np

# Load trained model and encoders
model = joblib.load("scheduler_model.pkl")
le_country = joblib.load("country_encoder.pkl")
le_content = joblib.load("content_encoder.pkl")


def format_hour(hour):
    if hour == 0:
        return "12 AM"
    elif hour < 12:
        return f"{hour} AM"
    elif hour == 12:
        return "12 PM"
    else:
        return f"{hour - 12} PM"


def ai_best_posting_times(country, content_type):
    country_num = le_country.transform([country])[0]
    content_num = le_content.transform([content_type])[0]

    hours = np.arange(24)

    X = np.column_stack([
        np.full(24, country_num),
        np.full(24, content_num),
        hours
    ])

    preds = model.predict(X)

    top_indices = np.argsort(preds)[::-1][:3]

    results = []
    for idx in top_indices:
        results.append((format_hour(int(idx)), int(preds[idx])))

    return results


# Streamlit UI
st.set_page_config(page_title="AI Global Marketing Scheduler", page_icon="🌍")

st.title("🌍 AI Global Marketing Scheduler")
st.write("Predict the best posting times for international audiences.")

working_country = st.selectbox(
    "Select Your Working Country",
    list(le_country.classes_)
)

target_country = st.selectbox(
    "Select Target Audience Country",
    list(le_country.classes_)
)

content_type = st.selectbox(
    "Select Content Type",
    list(le_content.classes_)
)

if st.button("Find Best Posting Times"):
    top_times = ai_best_posting_times(target_country, content_type)

    st.success("AI scheduling completed successfully!")

    st.write(f"**Working Country:** {working_country}")
    st.write(f"**Target Audience:** {target_country}")
    st.write(f"**Content Type:** {content_type}")

    st.subheader("🔥 Top 3 Best Posting Times")

    for i, (time_str, score) in enumerate(top_times, start=1):
        st.write(f"{i}. {time_str} (AI Score: {score})")