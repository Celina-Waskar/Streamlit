pip install prophet
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Sales1.csv", parse_dates=["date"])
    return df

df = load_data()

df['total_sales'] = df['total_sales'].round(2)
df.drop(columns=["sales_15min"], inplace=True)

Q1 = df["total_sales"].quantile(0.25)
Q3 = df["total_sales"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df["total_sales"] >= lower_bound) & (df["total_sales"] <= upper_bound)]
df = df[['store_id', 'date', 'total_sales', 'sales_type']]

df["date"] = pd.to_datetime(df["date"])
cal = calendar()
holidays = cal.holidays(start=df["date"].min(), end=df["date"].max())
df["holiday"] = df["date"].isin(holidays).astype(int)

# Streamlit UI
st.title("Prophet Sales Forecasting App")

# Sidebar Inputs
store_id = st.selectbox("Select Store ID", df["store_id"].unique())
sales_type = st.selectbox("Select Sales Type", df["sales_type"].unique())
forecast_days = st.number_input("Forecast Days", min_value=7, max_value=365, value=30, step=7)

# Filter Data
df_filtered = df[(df["store_id"] == store_id) & (df["sales_type"] == sales_type)].copy()

# Sort by Date
dff_main = df_filtered.sort_values(by="date")

# Identify Exogenous Features
exogenous_features = [col for col in dff_main.columns if col not in ("store_id", "date", "total_sales", "sales_type")]
add_regressors = exogenous_features[:]

# Define Lag and Rolling Features
lag_days = [7, 14]
rolling_windows = [7, 14, 30]

# Create Rolling Features
for window in rolling_windows:
    dff_main[f'rolling_{window}'] = dff_main['total_sales'].shift(1).rolling(window=window).mean()

# Create Lag Features
for lag in lag_days:
    dff_main[f'lag_{lag}'] = dff_main['total_sales'].shift(lag)

# Drop NaN values after creating all features
dff_main.dropna(inplace=True)

# Prepare Data for Prophet
dff_main = dff_main.rename(columns={'date': 'ds', 'total_sales': 'y'})

# Initialize Prophet model
model = Prophet(seasonality_mode="additive", changepoint_prior_scale=0.004)

# Add Exogenous Features as Regressors
for reg in add_regressors:
    model.add_regressor(reg)

# Fit Model
model.fit(dff_main)

# Create Future Dataframe
future = model.make_future_dataframe(periods=forecast_days, freq='D')
future = future.merge(dff_main[['ds'] + add_regressors], on='ds', how='left').fillna(method='ffill')

# Forecast
forecast = model.predict(future)
forecast.rename(columns={'ds': 'date', 'yhat': 'total_sales'}, inplace=True)


# Display Results
st.subheader("📊 Forecast Results")
st.dataframe(forecast.tail(forecast_days)[['date', 'total_sales']].style.format({"total_sales": "${:,.2f}"}))

# Plot Forecast for Future Dates
future_forecast = forecast.tail(forecast_days)
fig = px.line(future_forecast, x='date', y='total_sales', title='Future Sales Forecast', 
              labels={'total_sales': 'Total Sales', 'date': 'Date'}, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)
