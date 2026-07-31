import gridstatus
import pandas as pd
import boto3
from io import StringIO
import time

# --- CONFIGURATION ---
# Add S3 bucket holding your data lake for project
BUCKET_NAME = "caiso-reliability-lake" 

def fetch_and_upload_data():
    print(" Starting Grid Data Ingestion")

    # 1. Initialize the CAISO Connection
    # Use gridstatus API to access CAISO portal
    iso = gridstatus.CAISO()

    # 2. Define the 'Stress Test' Period
    # We specifically want the July 2024 Heatwave (July 8 - July 12)
    # because the grid was under extreme pressure.
    start_date = "2024-07-08"
    end_date = "2024-07-12"

    print(f" Downloading data from {start_date} to {end_date}...")

    # 3. Fetch LOAD (Consumer Demand)
    # This tells us how much power California used every 5 minutes.
    print(" Fetching Load Data")
    df_load = iso.get_load(start=start_date, end=end_date)
    
    # 4. Fetch FUEL MIX (Generation Supply)
    # This tells us how that power was made (Solar, Wind, Gas, Nuclear).
    print(" Fetching Fuel Mix Data")
    df_mix = iso.get_fuel_mix(start=start_date, end=end_date)

    # 5. Merge Load and Fuel Mix Data on the 'Time' column
    # so we can see "At 5:00 PM, Load was X and Solar was Y".
    print(" Merging Datasets")
    
    # We sort both by time to ensure alignment
    df_load = df_load.sort_values('Time')
    df_mix = df_mix.sort_values('Time')

    # 'merge_asof' is a specialized join for time-series. 
    # It matches timestamps that are extremely close (nearest neighbor), 
    # handling slight reporting delays between systems.
    df_combined = pd.merge_asof(
        df_load, 
        df_mix, 
        on='Time', 
        direction='nearest',
        tolerance=pd.Timedelta("5min")
    )

    # 6. Upload to AWS S3
    print(f" Uploading to AWS S3 bucket: {BUCKET_NAME}")
    
    # Convert DataFrame to a CSV string buffer 
    csv_buffer = StringIO()
    df_combined.to_csv(csv_buffer, index=False)
    
    # Talk to S3
    s3_resource = boto3.resource('s3')
    
    # Save as 'raw/heatwave_2024.csv'
    s3_resource.Object(BUCKET_NAME, 'raw/heatwave_2024.csv').put(Body=csv_buffer.getvalue())

    print(" Success, Data is now in Data Lake.")

if __name__ == "__main__":
    fetch_and_upload_data()
