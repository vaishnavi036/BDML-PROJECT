import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("cleaned_food_delivery.csv")
df = df.drop_duplicates()

# Separate features and target
X = df.drop(columns=["Delivery_Time_min", "Order_ID"])
y = df["Delivery_Time_min"]

# Identify column types
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

# Preprocessing
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

# Build final pipeline
final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Train the model
final_model.fit(X_train, y_train)

# Save the model
joblib.dump(final_model, "delivery_model.pkl")

print("Final model trained and saved successfully!")
print("Model: Linear Regression")
print("Saved as: delivery_model.pkl")