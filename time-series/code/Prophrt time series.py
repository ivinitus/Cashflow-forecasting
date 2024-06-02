import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller, acf
from pandas.plotting import lag_plot
import warnings

warnings.filterwarnings('ignore')

DATA_PATH = '/content/cash_flow_data_ final.xlsx'
FEATURES = ['Cash Flow', 'Revenue/Sales', 'Total Income', 'Tax', 'Net Profit',
            'Accounts Receivable', 'Accounts Payable', 'Total Expenditure', 'Interest',
            'GDP Growth', 'Inflation/Interest Rate', 'Expenses', 'Cap. Expenditure',
            'Seasonality', 'Client Payment Trends', 'EBIT']

def load_data(path):
    data = pd.read_excel(path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.fillna(0, inplace=True)
    return data

def data_preprocessing(data):
    print("Data Types:\n", data.dtypes)
    print("Number of duplicates:", data.duplicated().sum())
    print("Data Description:\n", data.describe())

def visualize_features(data, features):
    f, ax = plt.subplots(nrows=len(features), ncols=1, figsize=(15, 40))
    for i, feature in enumerate(features):
        sns.lineplot(x=data['Date'], y=data[feature], ax=ax[i], color='dodgerblue')
        ax[i].set_title(f'Feature: {feature}', fontsize=14)
        ax[i].set_ylabel(feature, fontsize=14)
        ax[i].set_xlim([data['Date'].min(), data['Date'].max()])
    plt.tight_layout()
    plt.show()

def check_stationarity(series):
    result = adfuller(series.values, autolag='AIC')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')
    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")

def stationarity_check(data, features):
    for feature in features:
        print(f'Checking stationarity for {feature}')
        print('--------------')
        check_stationarity(data[feature])

def plot_acf_feature(data, feature, nlags=70):
    acf_values, confidence_interval = acf(data[feature], nlags=nlags, alpha=0.05)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(acf_values)), acf_values, width=0.4)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(y=-confidence_interval[1][0], color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(y=confidence_interval[1][0], color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel('Lag (in days)')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation Function (ACF) for {feature}')
    plt.show()

def plot_lag_plots(data, feature, max_lag=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 10), sharex=True, sharey=True, dpi=200)
    for i, ax in enumerate(axes.flatten()[:max_lag]):
        lag_plot(data[feature], lag=i+1, ax=ax, c='firebrick')
        ax.set_title('Lag ' + str(i+1))
    fig.suptitle(f'Lag Plots of {feature}', y=1.05)
    plt.show()

def plot_boxplots(data, features):
    fig = plt.figure(figsize=(15, 15))
    plt.title('Outliers', fontsize=18)
    for i, col in enumerate(features):
        plt.subplot(8, 2, i+1)
        plt.title(col, fontsize=10)
        sns.boxplot(x=data[col])
        plt.axvline(data[col].mean(), linestyle='--', lw=4, zorder=1, color='red')
        plt.xlabel(col)
    plt.tight_layout()
    plt.show()

def find_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3 - q1
    outliers = df[((df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5 * IQR)))]
    return outliers

def remove_outliers(data, features):
    data_cleaned = data.copy()
    for col in features:
        outliers = find_outliers(data_cleaned[col])
        data_cleaned = data_cleaned[~data_cleaned[col].isin(outliers)]
    print("Data shape after removing outliers:", data_cleaned.shape)
    return data_cleaned

def forecast_with_prophet(data, feature):
    prophet_data = data[['Date', feature]].rename(columns={'Date': 'ds', feature: 'y'})
    model = Prophet()
    model.fit(prophet_data)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    fig = model.plot(forecast)
    plt.title(f'Forecast of {feature} using Prophet')
    plt.xlabel('Date')
    plt.ylabel(feature)
    plt.show()
    fig2 = model.plot_components(forecast)
    plt.show()
    print(f"Forecast data (last 12 months) for {feature}:\n", forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))
    return model, forecast

def svr_model(train, test, features):
    svr = SVR(kernel='linear')
    predictions = {}
    for feature in features:
        X_train = np.array(train.index).reshape(-1, 1)
        y_train = train[feature].values
        X_test = np.array(test.index).reshape(-1, 1)
        y_test = test[feature].values
        svr.fit(X_train, y_train)
        predictions[feature] = svr.predict(X_test)
    return predictions

def evaluate_predictions(test, predictions):
    for feature, preds in predictions.items():
        print(f"Evaluation for {feature}:")
        mse = mean_squared_error(test[feature], preds)
        mae = mean_absolute_error(test[feature], preds)
        mape = mean_absolute_percentage_error(test[feature], preds)
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print(f"MAPE: {mape}")

# Main Function
def main():
    data = load_data(DATA_PATH)
    data_preprocessing(data)
    visualize_features(data, FEATURES)
    stationarity_check(data, FEATURES)
    plot_acf_feature(data, 'Cash Flow')
    plot_lag_plots(data, 'Cash Flow')
    plot_boxplots(data, FEATURES)
    print(f'Number of outliers in Cash Flow: {len(find_outliers(data["Cash Flow"]))}')
    data_cleaned = remove_outliers(data, FEATURES)

    model, forecast = forecast_with_prophet(data, 'Cash Flow')
    for feature in FEATURES:
        forecast_with_prophet(data, feature)
    train, test = train_test_split(data_cleaned, test_size=0.2, shuffle=False)
    predictions = svr_model(train, test, FEATURES)
    evaluate_predictions(test, predictions)

if __name__ == "__main__":
    main()
