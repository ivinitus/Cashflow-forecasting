import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path, index_col='Date', parse_dates=True)
    data = data[~data.index.duplicated(keep='first')]
    data = data.asfreq('M')
    data['Cash Flow'] = data['Cash Flow'].interpolate()
    return data

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value}')

def plot_acf_pacf(timeseries):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(timeseries, ax=axes[0])
    plot_pacf(timeseries, ax=axes[1])
    plt.show()

def fit_arima_model(train, order):
    model = ARIMA(train['Cash Flow'], order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    model_fit.plot_diagnostics(figsize=(12, 8))
    plt.show()
    return model_fit

def plot_forecast(data, train, forecast_mean, conf_int, forecast_steps):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Cash Flow'], label='Historical')
    forecast_index = pd.date_range(start=train.index[-1], periods=forecast_steps+1, freq='M')[1:]
    plt.plot(forecast_index, forecast_mean, label='Forecast', linestyle='--')
    plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Cash Flow')
    plt.title('Cash Flow Forecast')
    plt.legend()
    plt.show()

def evaluate_forecast(test, forecast_mean):
    mae = mean_absolute_error(test['Cash Flow'], forecast_mean[:len(test)])
    mse = mean_squared_error(test['Cash Flow'], forecast_mean[:len(test)])
    epsilon = 1e-10
    mape = np.mean(np.abs((test['Cash Flow'] - forecast_mean[:len(test)]) / (test['Cash Flow'] + epsilon))) * 100
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'MAPE: {mape}%')

def main(file_path, forecast_steps=12, order=(1, 0, 1)):
    data = load_and_preprocess_data(file_path)

    print("Original Data Stationarity Check:")
    check_stationarity(data['Cash Flow'])

    plot_acf_pacf(data['Cash Flow'])

    train = data.iloc[:-forecast_steps]
    test = data.iloc[-forecast_steps:]

    model_fit = fit_arima_model(train, order)

    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    plot_forecast(data, train, forecast_mean, conf_int, forecast_steps)

    evaluate_forecast(test, forecast_mean)

if __name__ == "__main__":
    main('time.xlsx')
