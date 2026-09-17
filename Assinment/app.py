import joblib
import streamlit as st


@st.cache_resource
def load_pipeline():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_pipeline()

import pandas as pd

