import pandas as pd
import joblib
from flask import Flask, request, jsonify
from utils import get_stroke_risk_trend, ColumnSelector, FeatureEngineer

app = Flask(__name__)

# Load the pre-processing pipeline and the model from disk
pipeline = joblib.load("model_to_deploy/final_pipeline.pkl")

# To test if the server is up and running
@app.route("/test_server", methods=["GET"])
def test_server():
    return "OK - server is up and running!"

# To get the model's predictions
@app.route("/api/predict", methods=["POST"])
def predict():
    # Get JSON data from the request
    request_data = request.json

    # Extract the relevant fields and create a DataFrame
    # from the extracted data
    data = {
        "age": request_data["age"],
        "health_risk_score": request_data["health_risk_score"],
        "smoking_status": request_data["smoking_status"]
    }
    df = pd.DataFrame(data)

    # # Perform predictions using your pipeline (model)
    prediction = pipeline.predict(df)
    probability = pipeline.predict_proba(df)[:, 1] # Positive class probabilities

    # You can return the results as JSON
    return jsonify({
        "inputs": data,
        "prediction": prediction.tolist(),
        "stroke_probability": probability.tolist(),
    })


if __name__ == "__main__":
    app.run()
