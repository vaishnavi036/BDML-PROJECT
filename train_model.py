import pandas as pd

# Load the dataset
df = pd.read_csv("Food_Delivery_Times.csv")

# Display basic information
print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

# Handle missing categorical values
for col in ["Weather", "Traffic_Level", "Time_of_Day"]:
    df[col] = df[col].fillna(df[col].mode()[0])

# Handle missing numerical values
df["Courier_Experience_yrs"] = df["Courier_Experience_yrs"].fillna(
    df["Courier_Experience_yrs"].median()
)

# Remove duplicate records
df = df.drop_duplicates()

# Save the cleaned dataset
df.to_csv("cleaned_food_delivery.csv", index=False)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCleaned dataset saved successfully!")
print("Final Dataset Shape:", df.shape)