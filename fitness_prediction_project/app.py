import streamlit as st
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(
    page_title="Fitness & Calorie Predictor",
    page_icon="⚡",
    layout="centered"
)

# Load the Trained Model
@st.cache_resource
def load_model():
    with open('fitness_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# App Header
st.title("⚡ Fitness & Calorie Burn Predictor")
st.write("Input your profile and workout details below to predict total calories burned.")

st.markdown("---")

# User Input Form
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["male", "female"])
    age = st.number_input("Age (years)", min_value=10, max_value=100, value=25)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)

with col2:
    duration = st.number_input("Workout Duration (minutes)", min_value=1.0, max_value=300.0, value=30.0)
    heart_rate = st.number_input("Average Heart Rate (bpm)", min_value=60.0, max_value=220.0, value=105.0)
    body_temp = st.number_input("Body Temp (°C)", min_value=35.0, max_value=42.0, value=38.5)

st.markdown("---")

# Prediction Button
if st.button("🔥 Calculate Calorie Burn", use_container_width=True):
    # Construct DataFrame for Model Input
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Height': [height],
        'Weight': [weight],
        'Duration': [duration],
        'Heart_Rate': [heart_rate],
        'Body_Temp': [body_temp]
    })
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    # Calculate BMI
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    
    # Display Results
    st.success(f"### Estimated Calories Burned: **{prediction:.1f} kcal**")
    
    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    res_col1.metric(label="Body Mass Index (BMI)", value=f"{bmi:.1f}")
    
    if bmi < 18.5:
        bmi_status = "Underweight"
    elif 18.5 <= bmi < 25:
        bmi_status = "Normal weight"
    elif 25 <= bmi < 30:
        bmi_status = "Overweight"
    else:
        bmi_status = "Obese"
        
    res_col2.metric(label="BMI Category", value=bmi_status)