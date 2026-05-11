import pandas as pd
import time
import joblib
import os

# Paths to models and dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "datasets/predictive_maintenance.csv")
PIPELINE_PATH = os.path.join(BASE_DIR, "models/full_pipeline.pkl")
ANOMALY_PATH = os.path.join(BASE_DIR, "models/isolation_forest.pkl")

# Load Models
print("Loading models...")
pipeline = joblib.load(PIPELINE_PATH)
anomaly_model = joblib.load(ANOMALY_PATH)

# Load Dataset for simulation
df = pd.read_csv(DATA_PATH)

print("Starting Live Monitoring Simulation (Press Ctrl+C to stop)...")

for i in range(len(df)):
    sensor_row = df.iloc[i]
    
    # Prepare input for models
    # Pipeline expects: ['Type', 'Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
    # Encoding Type manually as we did in notebooks
    from sklearn.preprocessing import LabelEncoder
    input_data = sensor_row.to_frame().T
    
    # We need to ensure 'Type' is numeric as expected by the pipeline
    type_map = {'L': 1, 'M': 2, 'H': 0} # Approximate encoding based on notebook logic
    input_data['Type'] = input_data['Type'].map(type_map)
    
    # Select features for pipeline
    features = ['Type', 'Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
    input_features = input_data[features]
    
    # 1. Failure Prediction
    prediction = pipeline.predict(input_features)[0]
    probability = pipeline.predict_proba(input_features)[0][1]
    
    # 2. Anomaly Detection
    # Features for anomaly: ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
    anomaly_features = ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
    anomaly = anomaly_model.predict(input_data[anomaly_features])[0]
    
    # Monitoring Log
    status = "OK"
    if prediction == 1 or probability > 0.7:
        status = "!!! HIGH FAILURE RISK !!!"
    elif anomaly == -1:
        status = "??? ANOMALY DETECTED ???"
        
    print(f"Row {i} | Status: {status:25} | Prob: {probability:.4f} | Torque: {sensor_row['Torque']}")
    
    time.sleep(1) # Simulate 1 second delay between sensor readings
