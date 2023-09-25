import pandas as pd
import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.base import BaseEstimator, TransformerMixin

def get_stroke_risk_trend(age, base_prob=1, age_threshold=40):
    """Calculate so-called 'stroke risk trend': a function based on the age.

    Args:
        age (array-like): Age values.
        base_prob (float): Base probability of stroke (constant for
            age < age_threshold.)
        age_threshold (float): Age threshold after which the risk increases
            (doubles every 10 years).

    Returns:
        array-like: Stroke risk trend.
    """
    return np.where(
        age < age_threshold,
        base_prob,
        base_prob * 2 ** ((age - age_threshold) / 10),
    )

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Keeps only the indicated DataFrame columns
    and drops the rest.

    Attributes:
        feature_names (list): List of column names to keep.

    Methods:
        fit(X, y=None):
            Fit method (Returns self).
        transform(X):
            Transform method to select columns of interest.
            Returns a DataFrame with the selected columns only.
    """

    def __init__(self, keep):
        """Constructor

        Args.:
            keep (list): List of column names to keep.
        """
        self.keep = keep

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Select the indicated features from the input DataFrame X
        selected_features = X[self.keep]
        return pd.DataFrame(selected_features, columns=self.keep)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer to do feature engineering for the final model"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.assign(
            age_smoking_interaction=(
                X["age"] * (X["smoking_status"] != "never smoked")
            ),
            stroke_risk_40=get_stroke_risk_trend(X["age"], age_threshold=40),
        )

        cols_out = [
            "stroke_risk_40",
            "health_risk_score",
            "age_smoking_interaction",
        ]

        return X[cols_out]


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
    return ({
        "inputs": data,
        "prediction": prediction.tolist(),
        "stroke_probability": probability.tolist(),
    })

if __name__ == "__main__":
    app.run()
