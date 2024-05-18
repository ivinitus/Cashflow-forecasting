
"""This a simple regression with Split the data into training and testing sets

Original file is located at
    https://colab.research.google.com/drive/1xePZ5hVXI2gHjx8bb6CylTBB6Rjzfgr3
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import statsmodels.api as sm

# Define the dataset
df = pd.read_csv("reg.csv")

#  'Income', 'EBIT', and 'Sales' are the independent variables,
#  'Op Cash Flow' is the dependent variable 
X = df[['Income', 'EBIT', 'Sales']]
y = df['Op Cash Flow']

# Add a constant term to the independent variables matrix
X = sm.add_constant(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model using StatsModels
model = sm.OLS(y_train, X_train)
results = model.fit()

# Make predictions on the testing set
predictions = results.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Print the summary of the regression results
print(results.summary())
