import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import scipy.stats as stats

# Load data from CSV file
data = pd.read_csv('cash_flow_data_extended.csv')  # Replace 'data.csv' with your actual file path

# Column names based on provided features
features = [
    'Revenue/Sales', 'Total Income', 'Tax', 'Net Profit', 'Accounts Receivable',
    'Accounts Payable', 'EBIT', 'Total Expenditure', 'Interest',
    'GDP Growth', 'Inflation/Interest Rate', 'Expenses', 'Cap. Expenditure',
    'Seasonality', 'Client Payment Trends'
]
target = 'Cash Flow'

# Extract features and target
X = data[features]
y = data[target]

# Add a constant term for the intercept
X = sm.add_constant(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the OLS regression model
model = sm.OLS(y_train, X_train).fit()

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate residuals
residuals = y_test - y_pred

# Performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print model summary
print(model.summary())

# Plot the results
plt.figure(figsize=(14, 7))

# Actual vs Predicted Plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)  # Diagonal line
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
