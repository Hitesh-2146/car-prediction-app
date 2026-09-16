import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# ==============================================================================
# SECTION 1: PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Iris ML Classifier Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# SECTION 2: CUSTOM CSS STYLING FOR A CLEAN & ATTRACTIVE UI
# ==============================================================================
st.markdown("""
    <style>
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header Card styling */
    .header-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid #374151;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        color: #f9fafb;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .header-subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
    }
    
    /* Metric container styling */
    .metric-card {
        background-color: #1f2937;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #10b981;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 3: HEADER SECTION
# ==============================================================================
st.markdown("""
    <div class="header-card">
        <div class="header-title">🌸 Iris Species Classifier</div>
        <div class="header-subtitle">Real-time Machine Learning Inference Engine powered by Random Forest</div>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 4: MODEL & SCALER SETUP
# ==============================================================================
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

# ==============================================================================
# SECTION 5: SIDEBAR INPUT CONTROLS
# ==============================================================================
st.sidebar.markdown("### 🎛️ Feature Parameters")
st.sidebar.markdown("Adjust the flower measurements below to run prediction:")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, step=0.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, step=0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.35, step=0.05)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3, step=0.05)

input_df = pd.DataFrame({
    "Sepal Length (cm)": [sepal_length],
    "Sepal Width (cm)": [sepal_width],
    "Petal Length (cm)": [petal_length],
    "Petal Width (cm)": [petal_width]
})

# ==============================================================================
# SECTION 6: MAIN DASHBOARD DISPLAY
# ==============================================================================
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("📋 Input Summary")
    st.dataframe(input_df, use_container_width=True, hide_index=True)
    
    st.markdown("#### 📊 Quick Metrics")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric(label="Sepal Ratio", value=f"{(sepal_length/sepal_width):.2f}")
    with m_col2:
        st.metric(label="Petal Ratio", value=f"{(petal_length/petal_width):.2f}")

with col2:
    st.subheader("🎯 Model Inference Engine")
    
    if st.button("🚀 Predict Species", type="primary", use_container_width=True):
        scaled_input = scaler.transform(input_df.values)
        prediction = model.predict(scaled_input)[0]
        prediction_prob = model.predict_proba(scaled_input)[0]
        
        target_classes = ["Setosa", "Versicolor", "Virginica"]
        predicted_class = target_classes[prediction]
        confidence = prediction_prob[prediction] * 100
        
        st.success(f"**Predicted Species:** {predicted_class}")
        st.metric(label="Confidence Score", value=f"{confidence:.1f}%")
        
        st.markdown("---")
        st.markdown("#### 📈 Probability Distribution Across Classes")
        for cls, prob in zip(target_classes, prediction_prob):
            st.write(f"**{cls}** ({prob*100:.1f}%)")
            st.progress(float(prob))