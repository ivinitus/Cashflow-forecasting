import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(data, features, target):
    X = data[features]
    y = data[target]
    
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_scaled = sm.add_constant(X_scaled)
    
    return X_scaled, y

def check_multicollinearity(X):
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    vif["Feature"] = ['const'] + features
    print(vif)

def train_model(X_train, y_train):
    model = sm.OLS(y_train, X_train).fit()
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_squared_error')
    
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    print("Cross-Validation MSE:", -cv_scores.mean())
    print(model.summary())
    
    return y_pred, residuals

def plot_results(y_test, y_pred, residuals):
    plt.figure(figsize=(14, 7))

    # Actual vs Predicted Plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
    plt.xlabel('Actual Cash Flow')
    plt.ylabel('Predicted Cash Flow')
    plt.title('Actual vs Predicted Cash Flow')
    plt.grid(True)

    # Histogram of residuals
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=20, color='gray', edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')

    # Q-Q plot for normality
    plt.subplot(2, 2, 4)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # File path and column definitions
    filepath = 'cash_flow_data_extended.csv'  # Replace with your actual file path
    features = [
        'Revenue/Sales', 'Total Income', 'Tax', 'Net Profit', 'Accounts Receivable',
        'Accounts Payable', 'EBIT', 'Total Expenditure', 'Interest',
        'GDP Growth', 'Inflation/Interest Rate', 'Expenses', 'Cap. Expenditure',
        'Seasonality', 'Client Payment Trends'
    ]
    target = 'Cash Flow'
    
    # Load and preprocess data
    data = load_data(filepath)
    X_scaled, y = preprocess_data(data, features, target)
    
    # Check for multicollinearity
    check_multicollinearity(X_scaled)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    y_pred, residuals = evaluate_model(model, X_test, y_test)
    
    # Plot results
    plot_results(y_test, y_pred, residuals)
