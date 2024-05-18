import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.api import anova_lm


# Define the dataset
df = pd.read_csv("reg.csv")

# Let's assume 'Income', 'EBIT', and 'Sales' are the independent variables,
# and 'Op Cash Flow' is the dependent variable we want to forecast
X = df[['Income', 'EBIT', 'Sales']]
y = df['Op Cash Flow']

# Add a constant term to the independent variables matrix
X = sm.add_constant(X)

# Create and fit the linear regression model using StatsModels
model = sm.OLS(y, X)
results = model.fit()

# Print the summary of the model
print(results.summary())
