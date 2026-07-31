import pandas as pd
import boto3
import numpy as np
from io import StringIO

# --- CONFIGURATION ---
BUCKET_NAME = "caiso-reliability-lake"
INPUT_KEY = "raw/heatwave_2024.csv"
OUTPUT_KEY = "silver/processed_grid_data.csv"

def process_grid_data():
    print(" Initializing Feature Engineering Pipeline")

    # --- 1. DATA INGESTION ---
    # Load raw telemetry from S3 Data Lake
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=INPUT_KEY)
    df = pd.read_csv(obj['Body'])

    # Ensure temporal consistency: Parse timestamps and sort index
    # Critical for accurate rolling window calculations
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)

    print(f" Ingested {len(df)} records of raw telemetry.")

    # --- 2. DATA CLEANING ---
    # Imputation Strategy: Forward Fill (FFill), uses the last valid value to fill in missing values
    # Rationale: Grid sensors often drop packets. In time-series physics, 
    # assuming the state persists (ffill) is safer than 0-filling or interpolating 
    # across large gaps which can introduce artificial smoothing.
    df = df.ffill()

    # --- 3. FEATURE ENGINEERING: PHYSICS OF RELIABILITY ---
    
    # [A] Calculate Renewables Aggregate
    # Summation of intermittent resources to quantify total non-dispatchable generation
    # Note: Column names depend on gridstatus output; ensuring robustness for standard CAISO tags
    renewable_cols = [c for c in df.columns if 'Solar' in c or 'Wind' in c]
    df['Total_Renewables'] = df[renewable_cols].sum(axis=1)

    # [B] Net Load (The "Duck Curve" Metric)
    # Formula: Net Load = Total Demand - Renewable Generation
    # Significance: Represents the remaining load that MUST be met by thermal/hydro dispatch.
    # High Net Load = Stress on gas turbines. Low/Negative Net Load = Curtailment risk.
    df['Net_Load'] = df['Load'] - df['Total_Renewables']

    # [C] 15-Minute Ramp Rate (dNet_Load/dt)
    # Significance: Measures the velocity of net load change. 
    # Extreme positive ramps (> 200MW/min) risk frequency instability if reserves cannot respond fast enough.
    # We use .diff(3) because data is 5-min intervals (3 * 5 = 15 min window)
    df['Ramp_Rate_15min'] = df['Net_Load'].diff(3)

    # [D] Solar Volatility Index
    # Logic: 60-minute rolling standard deviation of Solar output
    # Description: The rolling SD captures how much renewable output bounces up and down over that hour.
    # Significance: High standard deviation implies cloud cover/intermittency.
    df['Solar_Volatility'] = df['Total_Renewables'].rolling(window=12).std()

    # --- 4. DATA QUALITY GATES ---
    # Defensibility: Drop the first 12 rows (NaNs from rolling windows) to prevent model artifacts
    df = df.dropna()

    print(" Feature Engineering Complete.")
    print(" Sample Metrics (Head):")
    print(df[['Time', 'Net_Load', 'Ramp_Rate_15min', 'Solar_Volatility']].head())

    # --- 5. LOADING (SAVE TO S3) ---
    # Persist the "Silver" dataset for downstream ML Consumption
    print(f" Uploading processed data to s3://{BUCKET_NAME}/{OUTPUT_KEY}...")
    
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=BUCKET_NAME, Key=OUTPUT_KEY, Body=csv_buffer.getvalue())

    print(" Pipeline Success. Silver dataset ready for Anomaly Detection.")

if __name__ == "__main__":
    process_grid_data()
