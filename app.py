from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("delivery_model.pkl")


# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- MODEL DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    try:
        df = pd.read_csv("model_comparison.csv")

        # Convert model comparison results into records
        models = df.to_dict(orient="records")

        # Find the model with the lowest MAE
        best_model = df.loc[df["MAE"].idxmin()].to_dict()

        return render_template(
            "dashboard.html",
            models=models,
            best_model=best_model
        )

    except FileNotFoundError:
        return "Error: model_comparison.csv was not found.", 404

    except (KeyError, ValueError) as error:
        app.logger.exception("Dashboard data error")
        return f"Error processing model comparison data: {error}", 500


# ---------------- DELIVERY PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect the user's input
        data = {
            "Distance_km": float(request.form["Distance_km"]),
            "Weather": request.form["Weather"],
            "Traffic_Level": request.form["Traffic_Level"],
            "Time_of_Day": request.form["Time_of_Day"],
            "Vehicle_Type": request.form["Vehicle_Type"],
            "Preparation_Time_min": float(
                request.form["Preparation_Time_min"]
            ),
            "Courier_Experience_yrs": float(
                request.form["Courier_Experience_yrs"]
            )
        }

        # Validate numeric inputs
        if (
            data["Distance_km"] <= 0
            or data["Preparation_Time_min"] < 0
            or data["Courier_Experience_yrs"] < 0
        ):
            return render_template(
                "index.html",
                error="Please enter a valid distance and non-negative preparation time and courier experience."
            ), 400

        # Arrange features in the correct order
        columns = [
            "Distance_km",
            "Weather",
            "Traffic_Level",
            "Time_of_Day",
            "Vehicle_Type",
            "Preparation_Time_min",
            "Courier_Experience_yrs"
        ]

        input_data = pd.DataFrame([data], columns=columns)

        # Generate the prediction
        prediction = float(model.predict(input_data)[0])
        prediction = max(0, prediction)

        # Generate helpful delivery instructions
        tips = []

        if data["Traffic_Level"] == "High":
            tips.append(
                "Traffic is heavy, so your delivery may take longer than usual."
            )

        if data["Weather"] in ["Rainy", "Stormy", "Snowy"]:
            tips.append(
                "Weather conditions may affect the courier's journey. "
                "Thank you for your patience."
            )

        if data["Distance_km"] > 10:
            tips.append(
                "Your destination is relatively far away. "
                "Please allow enough time for the journey."
            )

        if data["Preparation_Time_min"] > 20:
            tips.append(
                "Food preparation may take a little longer. "
                "Fresh food is worth the wait!"
            )

        if data["Time_of_Day"] in ["Evening", "Night"]:
            tips.append(
                "Delivery times can vary depending on road conditions "
                "and the time of day."
            )

        if not tips:
            tips.append(
                "Everything looks good! Keep your phone nearby "
                "for delivery updates."
            )

        # Show the result on a separate page
        return render_template(
            "result.html",
            prediction=round(prediction, 1),
            data=data,
            tips=tips
        )

    except (ValueError, TypeError, KeyError):
        return render_template(
            "index.html",
            error="Please check all your inputs and try again."
        ), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return render_template(
            "index.html",
            error=(
                "Sorry, we couldn't generate your prediction. "
                "Please try again."
            )
        ), 500


# ---------------- RUN APPLICATION ----------------
if __name__ == "__main__":
    app.run(debug=True)