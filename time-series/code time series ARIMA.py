import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the data
data = pd.read_excel('/content/time.xlsx')

# Convert the date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set the date column as the index
data.set_index('Date', inplace=True)

# Define and fit the ARIMA model
model = ARIMA(data['Cash Flow'], order=(5,1,0)) # Example order, you may need to tune this
model_fit = model.fit()

# Forecast future cash flows
forecast_steps = 12  # Adjust according to your needs
forecast = model_fit.forecast(steps=forecast_steps)

# Plot the original data and the forecast
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Cash Flow'], color='blue', label='Original')
plt.plot(pd.date_range(start=data.index[-1], periods=forecast_steps, freq='d'), forecast, color='red', label='Forecast')
plt.title('Cash Flow Forecasting')
plt.xlabel('Date')
plt.ylabel('Cash Flow')
plt.legend()
plt.show()

# Create a DataFrame for the forecasted values
forecast_dates = pd.date_range(start=data.index[-1], periods=forecast_steps, freq='d')
forecast_values = forecast
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Value': forecast_values})
print(forecast_df)

