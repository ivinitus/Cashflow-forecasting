import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(file_path):
    return pd.read_csv(file_path)

def split_data(data, features, target, test_size=0.2, random_state=42):
    X = data[features]
    y = data[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_linear_regression(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

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
    plt.title('Linear Regression: Actual vs Predicted')
    plt.show()

def main():
    file_path = '/content/cash_flow_data_ final.csv'
    data = load_data(file_path)

    features = [
        'Revenue/Sales', 'Total Income', 'Tax', 'Net Profit', 'Accounts Receivable',
        'Accounts Payable', 'Total Expenditure', 'Interest', 'GDP Growth',
        'Inflation/Interest Rate', 'Expenses', 'Cap. Expenditure', 'Seasonality',
        'Client Payment Trends', 'EBIT'
    ]
    target = 'Cash Flow'

    X_train, X_test, y_train, y_test = split_data(data, features, target)
    lr_model = train_linear_regression(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    mae_lr, mse_lr, r2_lr = evaluate_model(y_test, y_pred_lr)
    print(f'Linear Regression MAE: {mae_lr}')
    print(f'Linear Regression MSE: {mse_lr}')
    print(f'Linear Regression R-squared: {r2_lr}')

    results_lr = pd.DataFrame({
        'Actual Cash Flow': y_test,
        'Predicted Cash Flow (Linear Regression)': y_pred_lr
    })

    print(results_lr.head())

    plot_results(y_test, y_pred_lr)

if __name__ == "__main__":
    main()
