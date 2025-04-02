# Harveston-Climate-Prediction

# Weather Forecasting using Machine Learning & Deep Learning

## Overview
This project focuses on weather prediction using multiple models including SARIMAX, XGBoost, and statistical approaches. The dataset contains historical weather data, and the predictions include:
- **Avg Temperature (°C)**
- **Radiation (W/m²)**
- **Rain Amount (mm)**
- **Wind Speed (km/h)**
- **Wind Direction (°)**

## Models Used
### ✅ **1. Temperature Forecasting - SARIMAX**
- Handled missing values in `Avg_Temperature` using forward fill.
- Used **Seasonal ARIMA (SARIMAX)** to forecast temperature trends.
- Fallback to a mean-based trend prediction if SARIMAX fails.

### ✅ **2. Rainfall Prediction - XGBoost**
- Trained an **XGBoost Regressor** on month, year, and day as features.
- Handled missing values and replaced NaNs in `Rain_Amount` with 0.

### ✅ **3. Wind Direction Prediction**
- Used a **circular mean approach** for `Wind_Direction` to maintain direction consistency.
- Defaulted to `180°` (South) if no historical data was available.

### ✅ **4. Data Formatting & Cleaning**
- Ensured non-negative values for `Rain_Amount` and realistic values for `Wind_Speed` and `Radiation`.
- Generated reasonable values based on historical statistics.

### ✅ **5. Evaluation - sMAPE (Symmetric Mean Absolute Percentage Error)**
```python
import numpy as np

def smape(actual, predicted):
    """Calculate Symmetric Mean Absolute Percentage Error (sMAPE)."""
    mask = ~np.isnan(actual) & ~np.isnan(predicted)  # Remove NaN values
    actual, predicted = actual[mask], predicted[mask]
    
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))

# Compute sMAPE
N = len(test_df)
actual_values = train_df['Avg_Temperature'].iloc[-N:].values
predicted_values = test_df['Avg_Temperature'].values

actual_values = np.nan_to_num(actual_values, nan=np.nanmean(train_df['Avg_Temperature']))
predicted_values = np.nan_to_num(predicted_values, nan=np.nanmean(test_df['Avg_Temperature']))

smape_score = smape(actual_values, predicted_values)
print(f"sMAPE: {smape_score:.2f}%")
```

## Installation & Usage
1. **Clone the repository**
```bash
git clone https://github.com/SamsudeenAshad/Harveston-Climate-Prediction.git

```

## Results
- The model effectively predicts temperature, rainfall, wind speed, and direction.
- The **sMAPE score** is used to measure prediction accuracy.

## Contributions
Feel free to contribute by improving the model, fixing bugs, or adding new features.


