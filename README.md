# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.





## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the environmental sensor dataset and preprocess it by handling missing values and encoding categorical features. 2. Split the data into input features and target variables (temperature, PM2.5, energy), and normalize the features. 3. Divide the dataset into training and testing sets, then train a Random Forest Regressor on the training data. 4. Predict outputs on the test data and evaluate the model using metrics like MAE and R² score.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Swetha A
RegisterNumber: 212224040343<img width="1487" height="598" alt="564618325-83991528-ff42-48d7-9c41-c53c1fd4d3a5" src="https://github.com/user-attachments/assets/df1a5e6e-392a-4cad-ae5e-d52cb6f229d3" />
 
*/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
df.columns = df.columns.str.strip()


df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)


cols_to_fill = ['tem', 'pm2_5', 'tsr', 'hum', 'pressure', 'wind_speed', 'illumination', 'co2']
for col in cols_to_fill:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear', limit=10)


df['hour'] = df['time'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)


targets = ['tem', 'pm2_5', 'tsr']
for t in targets:
    df[f'{t}_lag1'] = df[t].shift(1)
    df[f'{t}_lag2'] = df[t].shift(2)


processed_df = df.dropna(subset=['tem_lag2', 'pm2_5_lag2', 'tsr_lag2', 'hum', 'pressure']).reset_index(drop=True)
processed_df.to_csv("combined_processed_weather_data.csv", index=False)


features = [
    'hum', 'pressure', 'wind_speed', 'illumination', 'co2',
    'hour_sin', 'hour_cos', 'tem_lag1', 'pm2_5_lag1', 'tsr_lag1'
]

print("--- Feature Engineering Summary ---")
print(f"Original rows: {len(df)}")
print(f"Processed rows (after lags/cleaning): {len(processed_df)}")
print(f"Final high-performance feature set:",features)

split_idx = int(len(processed_df) * 0.8)
train, test = processed_df.iloc[:split_idx], processed_df.iloc[split_idx:]
X_train, X_test = train[features], test[features]

models = {}
results = {}

target_meta = {
    'tem': ('Temperature', '°C', 'red'),
    'pm2_5': ('Pollution (PM2.5)', 'µg/m³', 'green'),
    'tsr': ('Energy (Solar Radiation)', 'W/m²', 'orange')
}

for target in targets:
    y_train, y_test = train[target], test[target]
    

    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    models[target] = model
    
    
    results[target] = {
        'r2': r2_score(y_test, preds),
        'mae': mean_absolute_error(y_test, preds),
        'preds': preds,
        'actual': y_test.values
    }


fig, axes = plt.subplots(3, 2, figsize=(16, 18))

for i, target in enumerate(targets):
    label, unit, color = target_meta[target]
    res = results[target]
    
   
    axes[i, 0].plot(res['actual'][-150:], label='Actual', color='black', alpha=0.4, linewidth=2)
    axes[i, 0].plot(res['preds'][-150:], label='Predicted', color=color, linestyle='--', linewidth=2)
    axes[i, 0].set_title(f"{label}: Actual vs Predicted\n$R^2$: {res['r2']:.3f} | MAE: {res['mae']:.2f}")
    axes[i, 0].set_ylabel(unit)
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)
    
   
    importances = pd.Series(models[target].feature_importances_, index=features).sort_values()
    importances.plot(kind='barh', ax=axes[i, 1], color=color, alpha=0.7)
    axes[i, 1].set_title(f"Key Drivers: {label}")

plt.tight_layout()
plt.show()


last_row = processed_df.iloc[-1]
latest_data = pd.DataFrame([{
    'hum': last_row['hum'], 'pressure': last_row['pressure'], 'wind_speed': last_row['wind_speed'],
    'illumination': last_row['illumination'], 'co2': last_row['co2'],
    'hour_sin': last_row['hour_sin'], 'hour_cos': last_row['hour_cos'],
    'tem_lag1': last_row['tem'], 'pm2_5_lag1': last_row['pm2_5'], 'tsr_lag1': last_row['tsr']
}])

print("\n--- NEXT STEP PREDICTIONS (Using Latest Data) ---")
for target in targets:
    pred_val = models[target].predict(latest_data)[0]
    print(f"Predicted {target_meta[target][0]}: {pred_val:.2f} {target_meta[target][1]}")


```

## Output:
<img width="1487" height="598" alt="564618325-83991528-ff42-48d7-9c41-c53c1fd4d3a5" src="https://github.com/user-attachments/assets/9641a27a-d492-445a-b2d5-96d02a1d1215" />
<img width="1444" height="545" alt="564618388-6ad8e7ef-b4cc-4af3-b8d2-6ca75b0a7f6c" src="https://github.com/user-attachments/assets/6e794095-202f-46d8-80c2-7d2c507409a0" />
<img width="1466" height="627" alt="564618434-935f8df4-ecea-4df2-b9f4-fc4f72cf328d" src="https://github.com/user-attachments/assets/7cbe4e4d-f5c8-421e-a110-1d84525a9318" />





## Result:
Thus,a python program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm has completed successfully.
