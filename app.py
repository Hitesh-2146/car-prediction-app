import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# Set up page configuration
st.set_page_config(page_title="ML Deployment Engine", page_icon="🤖", layout="wide")
st.title("🤖 End-to-End Streamlit ML Deployment Engine")
st.markdown("---")

# File paths for model artifacts
MODEL_PATH = "iris_rf_model.pkl"
SCALER_PATH = "iris_scaler.pkl"

def generate_mock_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        iris = load_iris()
        X, y = iris.data, iris.target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

generate_mock_artifacts()

@st.cache_resource
def load_ml_pipeline():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_ml_pipeline()

# Sidebar input widgets
st.sidebar.header("📥 Input Feature Values")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.35)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

input_df = pd.DataFrame({
    "sepal_length": [sepal_length],
    "sepal_width": [sepal_width],
    "petal_length": [petal_length],
    "petal_width": [petal_width]
})

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Selected Input Features")
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("🎯 Model Prediction Engine")
    if st.button("Run Model Inference", type="primary"):
        scaled_input = scaler.transform(input_df.values)
        prediction = model.predict(scaled_input)[0]
        prediction_prob = model.predict_proba(scaled_input)[0]
        
        target_classes = ["Setosa", "Versicolor", "Virginica"]
        predicted_class = target_classes[prediction]
        confidence = prediction_prob[prediction] * 100
        
        st.success(f"**Predicted Class:** {predicted_class}")
        st.metric(label="Prediction Confidence", value=f"{confidence:.2f}%")