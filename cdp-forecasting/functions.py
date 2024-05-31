import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# def time_series_prediction(data):
#     # Data preparation
#     # data = {
#     #     "XAxis": [
#     #         "2023-03-01", "2023-04-01", "2023-05-02", "2023-06-02", "2023-07-03",
#     #         "2023-08-03", "2023-09-03", "2023-10-04", "2023-11-04", "2023-12-05",
#     #         "2024-01-05", "2024-02-05", "2024-03-07", "2024-04-07", "2024-05-30"
#     #     ],
#     #     "YAxis": [
#     #         85008, 90521, 97096, 103172, 109138, 117920, 133261, 143840, 153799, 161191,
#     #         167172, 167173, 167176, 167181, 167181
#     #     ]
#     # }

#     df = pd.DataFrame(data)
#     df['XAxis'] = pd.to_datetime(df['XAxis'])
#     df.set_index('XAxis', inplace=True)

#     # Train the SARIMAX model
#     model = SARIMAX(df['YAxis'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
#     results = model.fit()

#     # Forecast future values
#     forecast_steps = 3
#     forecast = results.get_forecast(steps=forecast_steps)
#     forecast_df = forecast.conf_int()
#     forecast_df['ForecastedCount'] = forecast.predicted_mean

#     # Append forecast to the original data for visualization
#     forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='MS')[1:]
#     forecast_df.index = forecast_dates

#     df_with_forecast = pd.concat([df, forecast_df[['ForecastedCount']]], axis=1)

#     # Print forecasted values
#     # return(forecast_df[['ForecastedCount']])
#     return forecast_df


def time_series_prediction(data):
    df = pd.DataFrame(data)
    df['XAxis'] = pd.to_datetime(df['XAxis'])
    df.set_index('XAxis', inplace=True)

    # Train the SARIMAX model
    model = SARIMAX(df['YAxis'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()

    # Forecast future values
    forecast_steps = 3
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_df = forecast.conf_int()
    forecast_df['ForecastedCount'] = forecast.predicted_mean

    # Append forecast to the original data for visualization
    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='MS')[1:]
    forecast_df.index = forecast_dates

    # Combine original data and forecast data
    df_with_forecast = pd.concat([df, forecast_df[['ForecastedCount']]], axis=1)
    
    return df_with_forecast
