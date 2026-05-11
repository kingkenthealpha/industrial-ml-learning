from fastapi import FastAPI
import pandas as pd
import joblib
import os

# Create FastAPI app
app = FastAPI(title="Industrial ML Prediction API")

# Path to the saved pipeline
# Using absolute path logic to be safe, or relative to the file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/full_pipeline.pkl")
model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    """Welcome message for the API."""
    return {
        "message": "Industrial ML API Running",
        "status": "Healthy"
    }

@app.post("/predict")
def predict(data: dict):
    """
    Generate predictions for machine failure.
    Expected keys: Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear
    """
    try:
        # Convert input dictionary to DataFrame
        # The pipeline expects specific column names and order
        input_df = pd.DataFrame([data])
        
        # Ensure all required features are present
        required_features = ['Type', 'Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
        for feat in required_features:
            if feat not in input_df.columns:
                return {"error": f"Missing feature: {feat}"}

        # Reorder columns to match training data
        input_df = input_df[required_features]

        # Generate Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {
            "prediction": int(prediction),
            "failure_probability": float(probability),
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}
