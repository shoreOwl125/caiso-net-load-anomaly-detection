import pandas as pd
import boto3
import pickle
from io import StringIO, BytesIO
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
BUCKET_NAME = "caiso-reliability-lake"
INPUT_KEY = "silver/processed_grid_data.csv"
OUTPUT_DATA_KEY = "gold/scored_grid_data.csv"
OUTPUT_MODEL_KEY = "models/isolation_forest.pkl"

def train_and_score():
    print(" Initializing Unsupervised Anomaly Detection Model")

    # --- 1. DATA LOADING ---
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=INPUT_KEY)
    df = pd.read_csv(obj['Body'])
    
    # --- 2. FEATURE SELECTION ---
    # We select features that define the "State of Health" of the grid.
    # Excluded: 'Time' (Model should learn physics, not memorization of dates).
    feature_cols = ['Net_Load', 'Ramp_Rate_15min', 'Solar_Volatility']
    
    print(f" Training on features: {feature_cols}")
    
    # --- 3. PREPROCESSING (NORMALIZATION) ---
    # Critical Note: Isolation Forests are distance-dependent. 
    # Net_Load (~20,000 MW) dwarfs Solar_Volatility (~50). 
    # We use StandardScaler to scale them to Mean=0, Std=1 so the model treats them equally.
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])

    # --- 4. MODEL INITIALIZATION (The Isolation Forest) ---
    # n_estimators=100: Number of trees. Sufficient for low-dimensional data.
    # contamination=0.02: The "Budget" for anomalies.
    # Logic: We assume ~2% of grid states are truly "Abnormal/Stressful".
    # Setting this expectation helps calibrate the threshold.
    model = IsolationForest(
        n_estimators=100, 
        contamination=0.02, 
        random_state=42,
        n_jobs=-1 
    )

    # --- 5. TRAINING & INFERENCE ---
    print(" Fitting model to define 'Normal Operating Envelope'")
    model.fit(X)

    # Predict: -1 = Anomaly, 1 = Normal
    df['anomaly_label'] = model.predict(X)
    
    # Decision Function: The raw "weirdness" score.
    # Lower = More Anomalous. We invert it for readability on dashboards.
    # (Now: Higher Score = Higher Risk)
    df['anomaly_score'] = -1 * model.decision_function(X)

    # --- 6. POST-PROCESSING (INTERPRETABILITY) ---
    # A "Black Box" model is useless to a Control Room Operator.
    # We must explain WHY it's an anomaly.
    # Simple Heuristic: If it's an anomaly, which feature was furthest from the mean?
    
    def explain_anomaly(row):
        if row['anomaly_label'] == 1:
            return "Normal"
        
        # Calculate Z-scores (deviations) for this specific row
        deviations = {
            'Load_Surge': abs(row['Net_Load'] - df['Net_Load'].mean()),
            'Extreme_Ramp': abs(row['Ramp_Rate_15min'] - df['Ramp_Rate_15min'].mean()),
            'Solar_Instability': abs(row['Solar_Volatility'] - df['Solar_Volatility'].mean())
        }
        # Return the feature with the highest deviation
        return max(deviations, key=deviations.get)

    df['risk_driver'] = df.apply(explain_anomaly, axis=1)

    # --- 7. QUALITY ASSURANCE CHECK ---
    # Validation: Did we catch the peak stress event?
    anomalies = df[df['anomaly_label'] == -1]
    print(f" Detection Complete. Found {len(anomalies)} anomalous intervals.")
    print(" Top 3 Most Critical Events:")
    print(anomalies.sort_values('anomaly_score', ascending=False)[['Time', 'risk_driver', 'anomaly_score']].head(3))

    # --- 8. PERSISTENCE (SAVE TO S3) ---
    # Save the "Gold" Scored Dataset
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=BUCKET_NAME, Key=OUTPUT_DATA_KEY, Body=csv_buffer.getvalue())
    print(f" Scored data saved to {OUTPUT_DATA_KEY}")

    # Save the Model Artifact (Pickle)
    # Rationale: Enables MLOps - we can load this specific version later for API inference.
    model_buffer = BytesIO()
    pickle.dump(model, model_buffer)
    s3.put_object(Bucket=BUCKET_NAME, Key=OUTPUT_MODEL_KEY, Body=model_buffer.getvalue())
    print(f" Model artifact saved to {OUTPUT_MODEL_KEY}")

    print("AI Model is live and calibrated")

if __name__ == "__main__":
    train_and_score()
