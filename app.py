import streamlit as st
import pandas as pd
import boto3
import altair as alt
from io import BytesIO

# --- CONFIGURATION ---
# Page Config: Sets the browser tab title and layout to 'Wide' (Dashboard style)
st.set_page_config(page_title="CAISO Grid July 2024 Heatwave Inference Console", layout="wide")

BUCKET_NAME = "caiso-reliability-lake" # Update if your bucket name is different
DATA_KEY = "gold/scored_grid_data.csv"

# --- 1. DATA LOADER (With Caching) ---
# @st.cache_data prevents the app from re-downloading S3 data every time you click a button.
@st.cache_data
def load_data():
    # Helper to load data without crashing if keys are missing
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=DATA_KEY)
        df = pd.read_csv(obj['Body'])
        df['Time'] = pd.to_datetime(df['Time'])
        return df
    except Exception as e:
        return pd.DataFrame() # Return empty if fail

# Load the data once
df = load_data()

# Check if data loaded correctly
if df.empty:
    st.error(f"Could not load data from s3://{BUCKET_NAME}/{DATA_KEY}. Check your AWS credentials and bucket name.")
    st.stop()

# --- 2. SIDEBAR: THE CONTROLS ---
st.sidebar.header("Panel Controls")
st.sidebar.markdown("Filter grid telemetry to analyze stress events.")

# Date Filter
min_date = df['Time'].min().date()
max_date = df['Time'].max().date()
selected_date = st.sidebar.date_input("Select Operating Day", min_value=min_date, max_value=max_date, value=min_date)

# Filter Data by Date
day_df = df[df['Time'].dt.date == selected_date]

# Sensitivity Filter (Thresholding)
# Tuned Range: 0.0 to 0.05 allows you to find subtle anomalies.
risk_threshold = st.sidebar.slider(
    "Anomaly Confidence Threshold",
    min_value=0.000,
    max_value=0.050,
    value=0.000,
    step=0.005,
    format="%.3f",
)

# --- 3. MAIN DASHBOARD ---
st.title("CAISO Grid July 2024 Heatwave Inference Console")
st.markdown("### Event Replay & Anomaly Detection")

# Check if we have data for the selected day
if day_df.empty:
    st.warning("No data available for the selected date.")
    st.stop()

# --- REPLAY CONTROL (The "Time Machine") ---
# 1. Get all available timestamps for the selected day in HH:MM format
available_times = day_df['Time'].dt.strftime('%H:%M').unique()

# 2. The Slider
# This is the "Green Dot" control you asked for. 
# It lets you scrub through the day to see how the grid evolved.
selected_time_str = st.select_slider(
    "Playback Timeline (Drag to Replay Event)",
    options=available_times,
    value=available_times[-1] # Default to end of day (showing full history)
)

# 3. Filter Data (The "State at Time T")
# We filter the dataframe to only include data UP TO the selected time
subset_df = day_df[day_df['Time'].dt.strftime('%H:%M') <= selected_time_str]

# --- A. DYNAMIC KPI ROW ---
# Now these metrics reflect the "Live" state at the slider position
if not subset_df.empty:
    # Current Load (The last row in our subset)
    current_net_load = subset_df['Net_Load'].iloc[-1]
    
    # Ramp Rate (Current velocity)
    current_ramp = subset_df['Ramp_Rate_15min'].iloc[-1]
    
    # Cumulative Anomalies (How many red dots have appeared so far?)
    total_anomalies = subset_df[subset_df['anomaly_score'] > risk_threshold].shape[0]

    # Status Check: Is the CURRENT moment (or last 15 mins) critical?
    recent_risk = subset_df.tail(3)
    is_critical = not recent_risk[recent_risk['anomaly_score'] > risk_threshold].empty
    
    status_label = "CRITICAL ALERT" if is_critical else "NOMINAL"
    status_color = "inverse" if is_critical else "normal"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Net Load (Live)", f"{current_net_load:,.0f} MW", "Thermal Demand")
    col2.metric("Ramp Rate (15m)", f"{current_ramp:,.0f} MW", "Instant Velocity")
    col3.metric("System Status", status_label, delta_color=status_color)
    col4.metric("Anomalies Detected", f"{total_anomalies}", help="Cumulative count for selected playback window")

# --- B. THE CHART (Replay View) ---
# We plot the full day in GRAY (Forecast) and the playback in BLUE (Actual)

# 1. Background (The "Future" / Full Day Context)
base_chart = alt.Chart(day_df).encode(
    x=alt.X('Time:T', title='Time of Day')
).properties(height=400)

background_line = base_chart.mark_line(color='lightgray', strokeDash=[5,5]).encode(
    y='Net_Load:Q'
)

# 2. Active Replay Line (The "Past")
active_line = alt.Chart(subset_df).mark_line(color='#1f77b4').encode(
    x='Time:T',
    y=alt.Y('Net_Load:Q', title='Net Load (MW)'),
    tooltip=['Time', 'Net_Load', 'Ramp_Rate_15min', 'risk_driver']
)

# 3. Anomalies (Red Dots) - Only show if they happened in the past
active_anomalies = active_line.transform_filter(
    alt.datum.anomaly_score > risk_threshold
).mark_circle(size=100, color='red')

# 4. The "Current Time" Dot (Green)
# This sits at the very tip of the blue line
last_point = subset_df.iloc[[-1]] if not subset_df.empty else pd.DataFrame()

if not last_point.empty:
    current_dot = alt.Chart(last_point).mark_circle(size=200, color='green', opacity=1).encode(
        x='Time:T',
        y='Net_Load:Q',
        tooltip=['Time', 'Net_Load']
    )
else:
    current_dot = alt.Chart(pd.DataFrame()).mark_point()

# Combine: Background + Active Line + Red Anomalies + Green "You Are Here" Dot
final_chart = (background_line + active_line + active_anomalies + current_dot).interactive()

st.altair_chart(final_chart, use_container_width=True)

# --- C. DRILL DOWN: RISK DRIVERS ---
st.markdown("### Anomaly Diagnostics (Live Log)")

if not subset_df.empty:
    # Filter to only show the anomalies that have happened so far
    high_risk_df = subset_df[subset_df['anomaly_score'] > risk_threshold].copy()
    
    if not high_risk_df.empty:
        st.dataframe(
            high_risk_df[['Time', 'Net_Load', 'Ramp_Rate_15min', 'risk_driver', 'anomaly_score']]
            .sort_values('Time', ascending=False) # Show most recent events first
            .style.background_gradient(subset=['anomaly_score'], cmap='Reds'),
            use_container_width=True
        )
    else:
        st.success("✅ System Nominal. No anomalies detected in current playback window.")

# --- D. MODEL TECHNICAL SUMMARY (Footer) ---
st.markdown("---") # Visual divider line
st.subheader("Anomaly dectection was performed using an unsupervised Isolation Forest algorithm")

with st.expander("View Technical Details (Isolation Forest Specs)", expanded=False):
    st.markdown("""
    **Key Hyperparameters:**
    *   `n_estimators`: **100** (Number of base estimators in the ensemble).
    *   `contamination`: **Auto/0.02** (Calibrated strictly for high-confidence anomalies).
    *   `max_features`: **3** (Net Load, Ramp Rate, Solar Volatility).
    
    **Feature Engineering:**
    *   **Net Load (MW):** Calculated as `Total Load - (Solar + Wind)`. Captures the "Duck Curve" stress.
    *   **Ramp Rate (15m):** The first derivative of Net Load (`dLoad/dt`). Captures velocity stress on gas turbines.
    *   **Solar Volatility:** Rolling standard deviation (window=12) to detect intermittency/cloud cover.
    """)