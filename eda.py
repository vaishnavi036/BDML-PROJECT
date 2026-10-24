import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("cleaned_food_delivery.csv")

os.makedirs("graphs", exist_ok=True)

# Graph 1: Delivery Time Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Delivery_Time_min", kde=True)
plt.title("Delivery Time Distribution")
plt.tight_layout()
plt.savefig("graphs/delivery_distribution.png")
plt.close()

# Graph 2: Distance vs Delivery Time
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Distance_km", y="Delivery_Time_min")
plt.title("Distance vs Delivery Time")
plt.tight_layout()
plt.savefig("graphs/distance_vs_delivery.png")
plt.close()

# Graph 3: Traffic vs Delivery Time
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Traffic_Level", y="Delivery_Time_min")
plt.title("Traffic vs Delivery Time")
plt.tight_layout()
plt.savefig("graphs/traffic_vs_delivery.png")
plt.close()

# Graph 4: Weather vs Delivery Time
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Weather", y="Delivery_Time_min")
plt.title("Weather vs Delivery Time")
plt.tight_layout()
plt.savefig("graphs/weather_vs_delivery.png")
plt.close()

print("All graphs saved successfully in the graphs folder!")