import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv(r"G:\Py\Car predication\fitness_prediction_project\calories.csv")
print("Data Shape:", df.shape)
print("\nColumns found:", df.columns.tolist())

# 2. Separate Features (X) and Target (y)
feature_cols = ['Age', 'Gender', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
target_col = 'Calories'

X = df[feature_cols]
y = df[target_col]

# 3. Define Preprocessing
num_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
cat_features = ['Gender']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
    ]
)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Create and Train Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

print("\nTraining Model...")
pipeline.fit(X_train, y_train)

# 6. Model Evaluation
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} Calories")
print(f"R² Score: {r2:.4f}")

# 7. Save Model Pipeline
with open('fitness_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("\nModel saved successfully as 'fitness_model.pkl'!")