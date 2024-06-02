import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import scipy.stats as stats

def load_data(file_path):
    return pd.read_csv(file_path)

def get_features_and_target(data):
    features = [
        'Revenue/Sales', 'Total Income', 'Tax', 'Net Profit', 'Accounts Receivable',
        'Accounts Payable', 'EBIT', 'Total Expenditure', 'Interest',
        'GDP Growth', 'Inflation/Interest Rate', 'Expenses', 'Cap. Expenditure',
        'Seasonality', 'Client Payment Trends'
    ]
    target = 'Cash Flow'
    return data[features], data[target]

def add_intercept(X):
    return sm.add_constant(X)

def split_data(X, y, test_size=0.2, random_state=0):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_ols_model(X_train, y_train):
    model = sm.OLS(y_train, X_train).fit()
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    residuals = y_test - y_pred
    return y_pred, mse, r2, residuals

def plot_results(y_test, y_pred, residuals):
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
    plt.xlabel('Actual Cash Flow')
    plt.ylabel('Predicted Cash Flow')
    plt.title('Actual vs Predicted Cash Flow')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=20, color='gray', edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')

    plt.subplot(2, 2, 4)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')

    plt.tight_layout()
    plt.show()

def main():

    file_path = '/content/cash_flow_data_ final.csv'
    data = load_data(file_path)

    X, y = get_features_and_target(data)
    X = add_intercept(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_ols_model(X_train, y_train)

    print(model.summary())

    y_pred, mse, r2, residuals = evaluate_model(model, X_test, y_test)

    print(f'MSE: {mse}')
    print(f'R^2: {r2}')

    plot_results(y_test, y_pred, residuals)

if __name__ == "__main__":
    main()
