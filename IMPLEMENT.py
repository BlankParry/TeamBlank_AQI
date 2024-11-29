import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import folium
from streamlit_folium import st_folium

# Load Datasets
city_day = pd.read_csv('city_day.csv')
station_day = pd.read_csv('station_day.csv')

# Preprocessing Function
@st.cache_data
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2]
                                      else 'Summer' if x in [3, 4, 5]
                                      else 'Monsoon' if x in [6, 7, 8]
                                      else 'Post-Monsoon')
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

city_day = preprocess_data(city_day)
station_day = preprocess_data(station_day)

# Features and target
features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene']
target = 'AQI'

# Train-Test Split
@st.cache_data
def prepare_data(df):
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = prepare_data(city_day)

# Train the Model
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# Prediction and Evaluation
try:
    y_pred = model.predict(X_test)
    y_test = np.asarray(y_test, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
except Exception as e:
    st.error(f"Error during evaluation: {e}")
    rmse, r2 = None, None

# Streamlit Dashboard
st.title("Air Quality Prediction and Policy Simulation")
st.sidebar.header("Choose Analysis Type")

# Sidebar Options
analysis_type = st.sidebar.radio(
    "Select an option:",
    ["City-Level Analysis", "Station-Level Analysis", "AQI Prediction", "Policy Simulation", "Tree Plantation Impact"]
)

# City-Level Analysis
if analysis_type == "City-Level Analysis":
    st.header("City-Level Air Quality Analysis")
    city = st.selectbox("Choose a city:", city_day['City'].unique())
    city_data = city_day[city_day['City'] == city]
    st.subheader(f"Pollution Trends for {city}")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='Date', y='AQI', data=city_data, ax=ax, label='AQI')
    plt.title(f"AQI Trends in {city}")
    st.pyplot(fig)
    st.subheader("City Rankings by Average AQI")
    city_avg_aqi = city_day.groupby('City')['AQI'].mean().sort_values(ascending=False)
    st.table(city_avg_aqi)

# Station-Level Analysis
elif analysis_type == "Station-Level Analysis":
    st.header("Station-Level Air Quality Analysis")
    station_id = st.selectbox("Choose a station:", station_day['StationId'].unique())
    station_data = station_day[station_day['StationId'] == station_id]
    st.subheader(f"Pollution Trends for Station ID: {station_id}")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='Date', y='AQI', data=station_data, ax=ax, label='AQI')
    plt.title(f"AQI Trends at Station {station_id}")
    st.pyplot(fig)

# AQI Prediction
elif analysis_type == "AQI Prediction":
    st.header("Predict AQI")
    st.write(f"Model RMSE: {rmse:.2f}, R²: {r2:.2f}")
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"Enter {feature} level:", value=float(X_test.iloc[0][feature]))
    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.subheader(f"Predicted AQI: {prediction:.2f}")

# Policy Simulation
elif analysis_type == "Policy Simulation":
    st.header("Simulate Policy Changes")
    adjustments = {}
    for feature in features:
        adjustments[feature] = st.slider(f"Adjust {feature} (percent):", -50, 50, 0)
    if st.button("Simulate"):
        adjusted_data = X_test.iloc[0].copy()
        for feature, adj in adjustments.items():
            adjusted_data[feature] *= (1 + adj / 100)
        adjusted_prediction = model.predict(pd.DataFrame([adjusted_data]))[0]
        st.subheader(f"Adjusted Predicted AQI: {adjusted_prediction:.2f}")

# Tree Plantation Impact
elif analysis_type == "Tree Plantation Impact":
    st.header("Simulate Tree Plantation Impact")
    trees_planted = st.slider("Number of Trees Planted:", 0, 10000, step=500)
    reduction_factor = 0.02  # Example: 2% AQI improvement per 1000 trees
    reduction = trees_planted * reduction_factor
    if st.button("Simulate Tree Impact"):
        if len(y_test) > 0:
            st.write(f"Predicted AQI Reduction: {reduction:.2f}")
            st.write(f"Adjusted AQI: {max(0, y_test[0] - reduction):.2f}")
        else:
            st.write("Error: y_test is empty.")

# Heatmap Visualization
if st.checkbox("Show AQI Heatmap"):
    st.header("Geographic Heatmap of AQI")
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    for _, row in city_day.iterrows():
        folium.CircleMarker(
            location=[row.get('Latitude', 0), row.get('Longitude', 0)],
            radius=5,
            popup=f"{row['City']} AQI: {row['AQI']}",
            color='red' if row['AQI'] > 200 else 'green',
        ).add_to(m)
    st_folium(m)

# Footer
st.sidebar.markdown("Developed for Hackathons!")
