# -------------------------------
# TIME SERIES FORECASTING PROJECT
# USING XGBOOST
# -------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# 1. Load Dataset
df = pd.read_csv("PJME_hourly.csv")

# 2. Convert to datetime and set index
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# 3. Visualize Dataset
plt.figure(figsize=(10,4))
plt.plot(df.index, df['PJME_MW'])
plt.title("Energy Consumption Dataset")
plt.xlabel("Date")
plt.ylabel("MW Consumption")
plt.show()

# 4. Split data into train-test
train = df.loc[df.index < "2015-01-01"]
test = df.loc[df.index >= "2015-01-01"]

# 5. Create Time Features
def create_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

train = create_features(train)
test = create_features(test)

# 6. X and Y
FEATURES = ['hour', 'dayofweek', 'month', 'quarter', 'year', 'dayofyear']
TARGET = 'PJME_MW'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

# 7. Train Model: XGBoost
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=3,
    objective='reg:squarederror'
)

model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          verbose=False)

# 8. Predictions
test['Predictions'] = model.predict(X_test)

# 9. Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, test['Predictions']))
print("RMSE:", rmse)

# 10. Plot Real vs Predicted
plt.figure(figsize=(12,5))
plt.plot(test.index, y_test, label="Actual")
plt.plot(test.index, test['Predictions'], label="Predicted")
plt.title("Actual vs Predicted Energy Consumption")
plt.legend()
plt.show()

# 11. Error Analysis
daily_error = (test['Predictions'] - y_test).resample("D").mean()
plt.figure(figsize=(10,4))
plt.plot(daily_error)
plt.title("Daily Prediction Error")
plt.xlabel("Date")
plt.ylabel("Error")
plt.show()
