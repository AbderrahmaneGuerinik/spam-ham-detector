import streamlit as st
import pandas as pd
import joblib
import custom_transformers

model = joblib.load('./models/xgboost.joblib')

st.title("Spam Vs. Ham messages")

message = st.text_area("Enter your message :", height=100)


def handle_click(message: str):
    df = pd.DataFrame([{
        'v1': 'ham',
        'v2': message,
        'Unnamed: 2': None,
        'Unnamed: 3': None,
        'Unnamed: 4': None,
    }])
    prediction = model.predict(df)
    return prediction[0]


if st.button("Predict", type="primary", use_container_width=True):
    result = handle_click(message)

    if result == 1:
        st.error("🚨 This message is **SPAM**.")
    else:
        st.success("✅ This message is **HAM** (not spam).")
