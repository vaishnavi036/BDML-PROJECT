import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("cleaned_food_delivery.csv")
df = df.drop_duplicates()

# Separate features and target
X = df.drop(columns=["Delivery_Time_min", "Order_ID"])
y = df["Delivery_Time_min"]

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

# Preprocessing pipelines
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

# Hyperparameter combinations
params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}

# Grid search with cross-validation
grid = GridSearchCV(
    pipeline,
    params,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

print("Tuning model... Please wait.")
grid.fit(X_train, y_train)

# Evaluate the tuned model
best_model = grid.best_estimator_
predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nBest Parameters:", grid.best_params_)
print("Tuned Random Forest Results")
print(f"MAE:  {mae:.2f} minutes")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")

# Save tuned model
joblib.dump(best_model, "tuned_random_forest.pkl")

print("\nTuned model saved successfully!")