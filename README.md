# Seismic Time Series Modeling

Hybrid ARIMA–LSTM time series forecasting framework for monthly earthquake magnitude prediction with statistical (IQR, Z-score) and residual-based anomaly detection.

## Problem Overview

Earthquakes cause catastrophic loss of life and infrastructure damage with minimal warning.  
Physics-based prediction is difficult due to complex tectonic interactions and limited observability, but historical seismic records contain temporal patterns that are well-suited to time series modeling.  
This project focuses on **monthly magnitude forecasting** to provide actionable insights for disaster preparedness and risk assessment.

## Objectives

- **Preprocessing pipeline**: Transform raw event-level seismic data into a clean monthly time series with engineered temporal features.
- **Model comparison**: Benchmark a classical ARIMA baseline against deep learning LSTM models.
- **Hybrid ensemble**: Build a 70% LSTM + 30% ARIMA hybrid to leverage both nonlinear representation power and linear interpretability.
- **Anomaly detection**: Detect unusual seismic periods using IQR, Z-score, and residual-based methods.

## Methodology

### Data Processing

- Exploratory data analysis (EDA) on historical earthquake records.
- Monthly aggregation of magnitudes (and related features such as depth).
- Feature engineering for temporal structure (lags, rolling windows, etc.).

### ARIMA Modeling

- Autoregressive Integrated Moving Average for **linear** time series dynamics.
- Grid search over \((p, d, q)\) guided by Akaike Information Criterion (AIC).
- 80/20 train–test split preserving temporal order.
- 12‑month ahead forecasting beyond the test period.

### LSTM Modeling

- Stacked LSTM architecture with 64 → 32 units and Dropout (0.2) regularization.
- Input features include:
  - `Magnitude_lag1`, `Magnitude`, `Magnitude_roll3_mean`
  - `Depth_lag1`, `Depth`
- Trained on 12‑month sequence windows using Adam optimizer and MSE loss.
- Early stopping with patience = 10 to prevent overfitting.

### Hybrid ARIMA–LSTM Ensemble

- Bidirectional LSTM (96 → 48 units) plus dense layers with Dropout (0.2–0.3).
- Enhanced feature set:
  - Multiple lags (up to 3), exponential weighted moving (EWM) averages (3, 6),
  - Volatility, depth, and seasonal sinusoidal features (sin/cos).
- Weighted ensemble:
  - **70% LSTM + 30% ARIMA** for balanced accuracy and robustness.
- Trained with Adam (learning rate 0.001), batch size 16, and early stopping (patience = 10).

## Performance

- **LSTM**:
  - Lowest prediction error metrics (e.g., RMSE, MAE).
- **Hybrid ARIMA–LSTM**:
  - Best overall fit with \(R^2 \approx 0.9695\).
- **ARIMA**:
  - Serves as an interpretable baseline for linear trends.

## Anomaly Detection

- **IQR Method**:
  - Uses bounds \(Q1 - 1.5 \times IQR\) and \(Q3 + 1.5 \times IQR\).
  - Detected **26** anomalous months with extreme seismic activity.
- **Z‑score Method**:
  - Flags values with \(|Z| > 2.5\).
  - Detected **18** anomalous periods.
- Together, these methods highlight unusual magnitude spikes and provide robust early warning signals.

## Key Findings

- Hybrid ARIMA–LSTM achieved **R² ≈ 0.9695**.
- LSTM achieved very low prediction errors (e.g., RMSE ≈ 0.0215, MAE ≈ 0.0160).
- Anomaly detection surfaced 26 (IQR) and 18 (Z‑score) unusual seismic periods.
- The integrated pipeline combines accurate forecasting with interpretable anomaly detection to support earthquake risk assessment.

## References

- AmanxAI (2020). *Earthquake Prediction Model with Machine Learning.*  
  `https://amanxai.com/2020/11/12/earthquake-prediction-model-with-machine-learning/`
- Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control.* Wiley.
- Hochreiter, S., & Schmidhuber, J. (1997). “Long Short-Term Memory.” *Neural Computation*, 9(8), 1735–1780.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.
- International Seismological Centre (ISC). ISC-GEM Global Instrumental Earthquake Catalogue.  
  `https://www.isc.ac.uk/iscgem`
