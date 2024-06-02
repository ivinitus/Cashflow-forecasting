import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint as sp_randint

def load_data(file_path):
    return pd.read_csv(file_path)

def define_features_and_target(data):
    features = [
        'Revenue/Sales', 'Total Income', 'Tax', 'Net Profit', 'Accounts Receivable',
        'Accounts Payable', 'Total Expenditure', 'Interest', 'GDP Growth',
        'Inflation/Interest Rate', 'Expenses', 'Cap. Expenditure', 'Seasonality',
        'Client Payment Trends', 'EBIT'
    ]
    target = 'Cash Flow'
    X = data[features]
    y = data[target]
    return X, y

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def perform_random_search(X_train, y_train):
    gbr = GradientBoostingRegressor(random_state=42)
    param_dist = {
        'n_estimators': sp_randint(100, 1000),
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None] + list(np.arange(3, 50, 3)),
        'min_samples_split': sp_randint(2, 20),
        'min_samples_leaf': sp_randint(1, 20),
        'learning_rate': np.linspace(0.01, 0.3, num=30),
        'subsample': np.linspace(0.6, 1.0, num=10)
    }
    random_search = RandomizedSearchCV(gbr, param_distributions=param_dist, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

def plot_results(y_test, y_pred):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Gradient Boosting: Actual vs Predicted')
    plt.show()

def main():
    file_path = '/content/cash_flow_data_ final.csv'
    data = load_data(file_path)
    X, y = define_features_and_target(data)
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(X, y)

    best_gbr = perform_random_search(X_train_scaled, y_train)
    y_pred_gbr = best_gbr.predict(X_test_scaled)

    mae_gbr, mse_gbr, r2_gbr = evaluate_model(y_test, y_pred_gbr)
    print(f'Gradient Boosting MAE: {mae_gbr}')
    print(f'Gradient Boosting MSE: {mse_gbr}')
    print(f'Gradient Boosting R-squared: {r2_gbr}')

    results_gbr = pd.DataFrame({
        'Actual Cash Flow': y_test,
        'Predicted Cash Flow (Gradient Boosting)': y_pred_gbr
    })

    print(results_gbr.head())

    plot_results(y_test, y_pred_gbr)

if __name__ == "__main__":
    main()
