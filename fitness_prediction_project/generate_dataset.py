import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)
n_samples = 1000

# 1. Generate Input Features
age = np.random.randint(18, 65, size=n_samples)
gender = np.random.choice(["Male", "Female"], size=n_samples)
height_cm = np.round(np.random.normal(168, 10, size=n_samples), 1)  # Average height
weight_kg = np.round(np.random.normal(70, 15, size=n_samples), 1)  # Average weight

activity_level = np.random.choice(
    ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"],
    size=n_samples,
    p=[0.3, 0.4, 0.2, 0.1],
)

goal = np.random.choice(
    ["Weight Loss", "Maintain Weight", "Muscle Gain"],
    size=n_samples,
    p=[0.5, 0.3, 0.2],
)

# 2. Calculate Physiological Targets (Rule-Based Base Data)
bmi = weight_kg / ((height_cm / 100) ** 2)

# BMR via Mifflin-St Jeor Formula
bmr = np.where(
    gender == "Male",
    (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5,
    (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161,
)

# Activity Multipliers
activity_multipliers = {
    "Sedentary": 1.2,
    "Lightly Active": 1.375,
    "Moderately Active": 1.55,
    "Very Active": 1.725,
}
tdee_base = bmr * np.vectorize(activity_multipliers.get)(activity_level)

# Goal Caloric Adjustments
goal_adjustments = {
    "Weight Loss": -400,
    "Maintain Weight": 0,
    "Muscle Gain": 300,
}
target_calories = (
    tdee_base + np.vectorize(goal_adjustments.get)(goal)
) + np.random.normal(0, 50, n_samples)

# Daily Steps Calculation Strategy
base_steps = np.where(
    goal == "Weight Loss", 10000, np.where(goal == "Muscle Gain", 7500, 8500)
)
act_step_bonus = np.vectorize(
    {"Sedentary": 0, "Lightly Active": 1500, "Moderately Active": 3000, "Very Active": 5000}.get
)(activity_level)

daily_steps = (
    base_steps + act_step_bonus + np.random.normal(0, 500, n_samples)
).astype(int)
daily_water_liters = np.round((weight_kg * 0.033) + np.random.normal(0, 0.2, n_samples), 2)

# 3. Create DataFrame & Save
df = pd.DataFrame(
    {
        "Age": age,
        "Gender": gender,
        "Height_cm": height_cm,
        "Weight_kg": weight_kg,
        "Activity_Level": activity_level,
        "Goal": goal,
        "BMI": np.round(bmi, 2),
        "Daily_Steps_Target": daily_steps,
        "Target_Calories": np.round(target_calories, 0),
        "Daily_Water_Liters": daily_water_liters,
    }
)

df.to_csv("health_fitness_dataset.csv", index=False)
print("Dataset created successfully with shape:", df.shape)
print(df.head())
