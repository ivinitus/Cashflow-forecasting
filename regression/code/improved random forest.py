import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint as sp_randint

data = pd.read_csv('/content/cash_flow_data_ final.csv')

features = ['Revenue/Sales', 'Total Income', 'Tax', 'Net Profit', 'Accounts Receivable',
            'Accounts Payable', 'Total Expenditure', 'Interest', 'GDP Growth',
            'Inflation/Interest Rate', 'Expenses', 'Cap. Expenditure', 'Seasonality',
            'Client Payment Trends', 'EBIT']
target = 'Cash Flow'

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestRegressor(random_state=42)

param_dist = {
    'n_estimators': sp_randint(100, 1000),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None] + list(np.arange(10, 110, 10)),
    'min_samples_split': sp_randint(2, 20),
    'min_samples_leaf': sp_randint(1, 20),
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train_scaled, y_train)

best_rf = random_search.best_estimator_

best_rf.fit(X_train_scaled, y_train)
y_pred_rf = best_rf.predict(X_test_scaled)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'Improved Random Forest MAE: {mae_rf}')
print(f'Improved Random Forest MSE: {mse_rf}')
print(f'Improved Random Forest R-squared: {r2_rf}')

results_rf = pd.DataFrame({
    'Actual Cash Flow': y_test,
    'Predicted Cash Flow (Random Forest)': y_pred_rf
})

print(results_rf.head())

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Improved Random Forest: Actual vs Predicted')
plt.show()
