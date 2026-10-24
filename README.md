# 🍔 Foodie - Food Delivery Time Prediction

A machine learning web application that predicts the estimated time required to deliver food based on delivery distance, weather conditions, traffic level, time of day, vehicle type, food preparation time, and courier experience.

The application provides an interactive interface where users can enter delivery details and receive an estimated delivery time. It also includes a model performance dashboard to compare machine learning models.

## 🌐 Live Demo

**Live Application:** https://food-delivery-time-prediction-xtkl.onrender.com

**GitHub Repository:** https://github.com/Sow-mya6/Food-Delivery-Time-Prediction

## 📌 Project Overview

Food delivery time can vary depending on several factors, including distance, traffic, weather, and food preparation time.

This project uses machine learning to estimate delivery duration based on these input features. The trained model is integrated into a Flask web application, allowing users to obtain predictions through a simple and user-friendly interface.

## ✨ Features

- 🍔 **Interactive User Interface:** A simple and attractive interface for entering delivery details.
- ⏱️ **Delivery Time Prediction:** Estimates delivery duration using a trained machine learning model.
- 🌦️ **Weather-Based Inputs:** Supports different weather conditions.
- 🚦 **Traffic Consideration:** Includes traffic level as a prediction feature.
- 🛵 **Vehicle Selection:** Accepts the delivery vehicle type.
- 👨‍🍳 **Preparation Time:** Considers the time required to prepare the food.
- 📊 **Model Performance Dashboard:** Displays comparisons between machine learning models.
- 📈 **Evaluation Metrics:** Uses MAE, RMSE, and R² score to evaluate model performance.
- 💡 **Delivery Tips:** Provides helpful messages based on the entered delivery conditions.

## 🛠️ Technologies Used

### Programming Language
- Python

### Machine Learning and Data Processing
- Scikit-learn
- Pandas
- NumPy
- Joblib

### Web Development
- Flask
- HTML
- CSS
- JavaScript
- Chart.js

### Deployment and Version Control
- Git
- GitHub
- Render

## 🤖 Machine Learning Models

The project compares the following machine learning models:

1. Linear Regression
2. Gradient Boosting Regressor
3. Random Forest Regressor

The models are evaluated using the following metrics:

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error; measures the average absolute difference between actual and predicted delivery times. |
| RMSE | Root Mean Squared Error; gives greater weight to larger prediction errors. |
| R² Score | Indicates how well the model explains variation in the target values. |

The model comparison dashboard presents these metrics to help evaluate model performance.

## 📊 Sample Predictions

The following predictions were obtained while testing the application with different input conditions.

| Test Case | Distance | Weather | Traffic | Preparation Time | Courier Experience | Predicted Delivery Time |
|---|---:|---|---|---:|---:|---:|
| Normal Conditions | 3 km | Clear | Low | 10 min | 3 years | 22.9 minutes |
| Heavy Traffic | 8 km | Rainy | High | 20 min | 1 year | 64.8 minutes |
| Long Distance | 15 km | Stormy | Medium | 30 min | 2 years | 87.5 minutes |

*These are sample predictions from the application, not a guarantee of actual delivery time.*

## 📁 Project Structure

```text
Food-Delivery-Time-Prediction/
│
├── app.py
├── delivery_model.pkl
├── tuned_random_forest.pkl
├── Food_Delivery_Times.csv
├── cleaned_food_delivery.csv
├── model_comparison.csv
│
├── eda.py
├── model_training.py
├── train_model.py
├── tune_model.py
├── save_final_model.py
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   ├── index.html
│   ├── result.html
│   └── dashboard.html
│
├── static/
│   └── style.css
│
└── graphs/
