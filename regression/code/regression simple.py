import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.api import anova_lm

df = pd.read_csv("reg.csv")
X = df[['Income', 'EBIT', 'Sales']]
y = df['Op Cash Flow']
X = sm.add_cons
model = sm.OLS(y, X)
results = model.fit()

print(results.summary())
