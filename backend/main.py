import os
import io
import json
import warnings
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pickle

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = FastAPI(title="Seismic Time Series Modeling API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for cached data and models
df = None
trained_models = {}
model_metrics = {}

def get_models_path():
    """Get the correct path to models directory based on current working directory"""
    if os.path.exists("backend/data/models"):
        return "backend/data/models"
    else:
        return "data/models"

def get_dataset_path():
    """Get the correct path to dataset based on current working directory"""
    if os.path.exists("backend/data/Dataset.csv"):
        return "backend/data/Dataset.csv"
    else:
        return "data/Dataset.csv"

def load_dataset():
    """Load and preprocess the dataset at startup"""
    global df
    try:
        dataset_path = get_dataset_path()
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        df = pd.read_csv(dataset_path)
        df.columns = df.columns.str.strip()
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        elif 'Date' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            df['Datetime'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        required_cols = ['Latitude', 'Longitude', 'Depth', 'Magnitude']
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.random.randn(len(df))
        df = df.dropna(subset=required_cols)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notnull(df), None)
        if len(df) < 10:
            raise ValueError("Dataset too small after cleaning")
        df = df.sort_values('Datetime').reset_index(drop=True)
        print(f"Dataset loaded successfully: {len(df)} rows")
        return True
    except Exception as e:
        print(f"Error loading dataset: {e}")
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Datetime': dates,
            'Latitude': np.random.uniform(35, 40, 100),
            'Longitude': np.random.uniform(-120, -115, 100),
            'Depth': np.random.uniform(0, 50, 100),
            'Magnitude': np.random.uniform(1, 7, 100)
        })
        print("Using dummy dataset")
        return False

@app.on_event("startup")
async def startup_event():
    # Create models directory - use correct path based on working directory
    models_dir = get_models_path()
    os.makedirs(models_dir, exist_ok=True)
    load_dataset()

@app.get("/")
async def root():
    return {
        "message": "Seismic Time Series Modeling API",
        "status": "running",
        "endpoints": {
            "data": "/api/data/head",
            "visuals": "/api/visuals/{plot_type}",
            "train": "/api/train",
            "anomalies": "/api/anomalies",
            "anomaly_detection": "/api/anomaly"
        }
    }

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon available"}

@app.get("/api/data/head")
async def get_data_head():
    try:
        if df is None:
            raise HTTPException(status_code=500, detail="Dataset not loaded")
        head_data = df.head(10).copy()
        if 'Datetime' in head_data.columns:
            head_data['Datetime'] = head_data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        def clean_value(val):
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                return None
            return val
        data_cleaned = []
        for record in head_data.to_dict('records'):
            data_cleaned.append({k: clean_value(v) for k, v in record.items()})
        return {
            "data": data_cleaned,
            "columns": list(head_data.columns),
            "total_rows": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")

def create_plot_response(fig):
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close(fig)
    return StreamingResponse(img_buffer, media_type="image/png")

@app.get("/api/visuals/{plot_type}")
async def get_visualization(plot_type: str):
    try:
        if df is None:
            raise HTTPException(status_code=500, detail="Dataset not loaded")
        plt.style.use('default')
        if plot_type == "magnitude_hist":
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['Magnitude'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Magnitude')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Earthquake Magnitudes')
            ax.grid(True, alpha=0.3)
        elif plot_type == "depth_hist":
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['Depth'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            ax.set_xlabel('Depth (km)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Earthquake Depths')
            ax.grid(True, alpha=0.3)
        elif plot_type == "corr_heatmap":
            fig, ax = plt.subplots(figsize=(10, 8))
            numeric_cols = ['Latitude', 'Longitude', 'Depth', 'Magnitude']
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Matrix of Seismic Features')
        elif plot_type == "scatter_locations":
            fig, ax = plt.subplots(figsize=(12, 8))
            scatter = ax.scatter(df['Longitude'], df['Latitude'], 
                               c=df['Magnitude'], s=df['Depth']*2, 
                               alpha=0.6, cmap='viridis')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('Earthquake Locations (Color: Magnitude, Size: Depth)')
            plt.colorbar(scatter, label='Magnitude')
        elif plot_type == "magnitude_time":
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(df['Datetime'], df['Magnitude'], alpha=0.7, linewidth=1)
            ax.set_xlabel('Date')
            ax.set_ylabel('Magnitude')
            ax.set_title('Magnitude Over Time')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        elif plot_type == "monthly_boxplot":
            fig, ax = plt.subplots(figsize=(12, 6))
            df_copy = df.copy()
            df_copy['Month'] = df_copy['Datetime'].dt.month
            monthly_data = [df_copy[df_copy['Month'] == month]['Magnitude'].values 
                          for month in range(1, 13)]
            ax.boxplot(monthly_data, labels=[f'M{i}' for i in range(1, 13)])
            ax.set_xlabel('Month')
            ax.set_ylabel('Magnitude')
            ax.set_title('Monthly Distribution of Magnitudes')
            ax.grid(True, alpha=0.3)
        else:
            raise HTTPException(status_code=404, detail="Plot type not found")
        return create_plot_response(fig)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating plot: {str(e)}")

@app.get("/api/anomalies/plot")
async def get_anomaly_plot():
    try:
        if df is None or not trained_models:
            raise HTTPException(status_code=400, detail="Dataset or models not available")
        
        # Find best model based on lowest RMSE
        best_model = min(model_metrics.keys(), key=lambda k: model_metrics[k]['RMSE'])
        predictions = trained_models[best_model]['predictions']
        actual = df['Magnitude'].values
        
        # Ensure arrays have the same length
        min_len = min(len(predictions), len(actual))
        predictions = predictions[:min_len]
        actual = actual[:min_len]
        dates = df['Datetime'].iloc[:min_len]
        
        if len(predictions) == 0:
            raise HTTPException(status_code=400, detail="No valid predictions available")
        
        residuals = actual - predictions
        
        # Calculate anomalies
        residual_std = np.std(residuals)
        if residual_std == 0:
            # No anomalies if no variation
            z_scores = np.zeros_like(residuals)
        else:
            z_scores = np.abs((residuals - np.mean(residuals)) / residual_std)
        
        threshold = 2.5
        anomaly_mask = z_scores > threshold
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top plot: Actual vs Predicted with anomalies highlighted
        ax1.plot(dates, actual, label='Actual', alpha=0.7, linewidth=1)
        ax1.plot(dates, predictions, label='Predicted', alpha=0.7, linewidth=1)
        
        # Highlight anomalies
        anomaly_dates = dates[anomaly_mask]
        anomaly_actual = actual[anomaly_mask]
        ax1.scatter(anomaly_dates, anomaly_actual, color='red', s=50, alpha=0.8, 
                   label=f'Anomalies ({np.sum(anomaly_mask)})', zorder=5)
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Magnitude')
        ax1.set_title(f'Earthquake Magnitude: Actual vs Predicted ({best_model} Model)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Bottom plot: Residuals with anomaly threshold
        ax2.scatter(dates, residuals, alpha=0.6, s=20, c='blue', label='Residuals')
        ax2.scatter(anomaly_dates, residuals[anomaly_mask], color='red', s=50, 
                   alpha=0.8, label='Anomalies', zorder=5)
        
        # Add threshold lines
        mean_residual = np.mean(residuals)
        threshold_upper = mean_residual + threshold * residual_std
        threshold_lower = mean_residual - threshold * residual_std
        
        ax2.axhline(y=threshold_upper, color='red', linestyle='--', alpha=0.7, 
                   label=f'Threshold (±{threshold}σ)')
        ax2.axhline(y=threshold_lower, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=mean_residual, color='green', linestyle='-', alpha=0.7, 
                   label='Mean Residual')
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Residual (Actual - Predicted)')
        ax2.set_title('Residuals and Anomaly Detection')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return create_plot_response(fig)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating anomaly plot: {str(e)}")

def calculate_metrics(y_true, y_pred):
    try:
        # Convert to numpy arrays
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        
        # Ensure y_true and y_pred are the same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[-min_len:]
        y_pred = y_pred[-min_len:]
        
        # Handle NaN and infinite values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {"R2": 0, "MAE": 0, "RMSE": 0, "MAPE": 0, "sMAPE": 0}
        
        # Calculate metrics
        r2 = r2_score(y_true_clean, y_pred_clean)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        
        # Calculate MAPE with better handling of zero values
        epsilon = 1e-8
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / np.maximum(np.abs(y_true_clean), epsilon))) * 100
        
        # Calculate sMAPE properly
        smape = np.mean(2 * np.abs(y_true_clean - y_pred_clean) / 
                       np.maximum(np.abs(y_true_clean) + np.abs(y_pred_clean), epsilon)) * 100
        
        # Ensure all metrics are reasonable
        r2 = max(-1, min(1, r2))  # R2 should be between -1 and 1
        mape = min(mape, 1000)    # Cap MAPE at 1000%
        smape = min(smape, 200)   # Cap sMAPE at 200%
        
        return {
            "R2": float(r2),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "sMAPE": float(smape)
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        print(f"y_true shape: {np.asarray(y_true).shape}, y_pred shape: {np.asarray(y_pred).shape}")
        print(f"y_true sample: {np.asarray(y_true)[:5]}")
        print(f"y_pred sample: {np.asarray(y_pred)[:5]}")
        return {"R2": 0, "MAE": 0, "RMSE": 0, "MAPE": 0, "sMAPE": 0}

def train_arima_model(data, order=(1,1,1)):
    try:
        model = ARIMA(data, order=order)
        fitted_model = model.fit()
        predictions = fitted_model.fittedvalues
        
        # Ensure predictions match data length
        if len(predictions) < len(data):
            # Pad with first available prediction value
            pad_value = predictions[0] if len(predictions) > 0 else np.mean(data)
            pad_len = len(data) - len(predictions)
            predictions = np.concatenate([np.full(pad_len, pad_value), predictions])
        elif len(predictions) > len(data):
            predictions = predictions[-len(data):]
            
        return fitted_model, predictions
    except Exception as e:
        print(f"ARIMA training error: {e}")
        return None, np.full(len(data), np.mean(data))

def train_lstm_model(data, lookback=10):
    try:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i])
        if len(X) == 0:
            return None, None, np.full(len(data), np.mean(data))
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        
        # Get predictions and transform back to original scale
        scaled_predictions = model.predict(X, verbose=0).flatten()
        predictions = scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).flatten()
        
        # Pad with the mean of the first lookback points (not NaN)
        pad_value = np.mean(data[:lookback])
        pad_len = len(data) - len(predictions)
        if pad_len > 0:
            predictions = np.concatenate([np.full(pad_len, pad_value), predictions])
        elif pad_len < 0:
            predictions = predictions[-len(data):]
            
        return model, scaler, predictions
    except Exception as e:
        print(f"LSTM training error: {e}")
        return None, None, np.full(len(data), np.mean(data))

@app.post("/api/train")
async def train_models(request: Request):
    def match_length(arr, target_len):
        arr = np.asarray(arr)
        if len(arr) < target_len:
            arr = np.concatenate([np.full(target_len - len(arr), np.nan), arr])
        elif len(arr) > target_len:
            arr = arr[-target_len:]
        return arr
    
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    global trained_models, model_metrics
    import glob
    models_dir = get_models_path()
    model_files = sorted(glob.glob(f"{models_dir}/models_*.pkl"), reverse=True)
    
    if model_files:
        try:
            with open(model_files[0], 'rb') as f:
                trained_models = pickle.load(f)
            # Try to load LSTM model from file if path exists
            if "LSTM" in trained_models and "model_path" in trained_models["LSTM"]:
                try:
                    from tensorflow.keras.models import load_model # type: ignore
                    model_path = trained_models["LSTM"]["model_path"]
                    
                    # Robust path resolution - check multiple possible locations
                    original_path = model_path
                    possible_paths = [
                        model_path,  # Try original path first
                        os.path.abspath(model_path),  # Try absolute version
                    ]
                    
                    # If path doesn't exist, try different combinations
                    if not os.path.exists(model_path):
                        filename = os.path.basename(model_path)
                        models_dir = get_models_path()
                        possible_paths.extend([
                            os.path.join(models_dir, filename),  # models_dir + filename
                            os.path.abspath(os.path.join(models_dir, filename)),  # absolute version
                            f"backend/data/models/{filename}",  # explicit backend path
                            f"data/models/{filename}",  # relative data path
                        ])
                    
                    # Find the first path that exists
                    working_path = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            working_path = path
                            break
                    
                    if working_path is None:
                        raise FileNotFoundError(f"LSTM model file not found. Tried paths: {possible_paths}")
                    
                    print(f"Loading LSTM model from: {working_path}")
                    lstm_model = load_model(working_path)
                    trained_models["LSTM"]["model"] = None  # Don't keep model object in dict
                except Exception as e:
                    print(f"Error loading LSTM model (version incompatibility): {e}")
                    print("Will retrain models...")
                    # Clear the incompatible models and retrain
                    trained_models = {}
                    model_files = []
            if "metrics" not in trained_models and trained_models:
                target_data = df['Magnitude'].values
                arima_pred = trained_models["ARIMA"]["predictions"]
                lstm_pred = trained_models["LSTM"]["predictions"]
                hybrid_pred = trained_models["Hybrid"]["predictions"]
                results = {}
                results["ARIMA"] = calculate_metrics(target_data, arima_pred)
                results["LSTM"] = calculate_metrics(target_data, lstm_pred)
                results["Hybrid"] = calculate_metrics(target_data, hybrid_pred)
                trained_models["metrics"] = results
                with open(model_files[0], 'wb') as f:
                    pickle.dump(trained_models, f)
            if trained_models and "metrics" in trained_models:
                model_metrics = trained_models["metrics"]
                return {
                    "status": "loaded",
                    "metrics": model_metrics,
                    "timestamp": model_files[0].split("models_")[-1].split(".pkl")[0] if model_files else timestamp,
                    "data_points": len(df['Magnitude'].values)
                }
        except Exception as e:
            print(f"Error loading existing models: {e}")
            print("Will retrain models...")
            trained_models = {}
            model_files = []
    
    target_data = df['Magnitude'].values
    if len(target_data) < 20:
        raise HTTPException(status_code=400, detail="Insufficient data for training")
    
    # Generate timestamp for this training session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {}
    print("Training ARIMA...")
    arima_model, arima_pred = train_arima_model(target_data)
    arima_pred = match_length(arima_pred, len(target_data))
    results["ARIMA"] = calculate_metrics(target_data, arima_pred)
    trained_models["ARIMA"] = {"model": arima_model, "predictions": arima_pred}
    
    print("Training LSTM...")
    lstm_model, lstm_scaler, lstm_pred = train_lstm_model(target_data)
    lstm_pred = match_length(lstm_pred, len(target_data))
    models_dir = get_models_path()
    lstm_model_path = f"{models_dir}/lstm_model_{timestamp}.keras"
    # Convert to absolute path to prevent resolution issues
    lstm_model_path = os.path.abspath(lstm_model_path)
    lstm_model.save(lstm_model_path)
    results["LSTM"] = calculate_metrics(target_data, lstm_pred)
    trained_models["LSTM"] = {"model_path": lstm_model_path, "scaler": lstm_scaler, "predictions": lstm_pred}
    
    print("Training Hybrid...")
    # Calculate residuals properly
    arima_residuals = target_data - arima_pred
    
    # Ensure residuals have the same length as target data
    if len(arima_residuals) != len(target_data):
        arima_residuals = match_length(arima_residuals, len(target_data))
    
    # Train LSTM on residuals
    _, _, lstm_residual_pred = train_lstm_model(arima_residuals)
    
    # Ensure residual predictions have correct length
    lstm_residual_pred = match_length(lstm_residual_pred, len(target_data))
    
    # Combine ARIMA and LSTM residual predictions
    hybrid_pred = arima_pred + lstm_residual_pred
    hybrid_pred = match_length(hybrid_pred, len(target_data))
    
    results["Hybrid"] = calculate_metrics(target_data, hybrid_pred)
    trained_models["Hybrid"] = {"predictions": hybrid_pred}
    
    trained_models["metrics"] = results
    models_dir = get_models_path()
    model_path = f"{models_dir}/models_{timestamp}.pkl"
    
    # Remove keras model object before pickling
    if "LSTM" in trained_models and "model" in trained_models["LSTM"]:
        trained_models["LSTM"]["model"] = None
    
    with open(model_path, 'wb') as f:
        pickle.dump(trained_models, f)
    
    model_metrics = results
    
    # Also update latest model file with metrics if it exists and is missing metrics
    models_dir = get_models_path()
    model_files = sorted(glob.glob(f"{models_dir}/models_*.pkl"), reverse=True)
    if model_files:
        with open(model_files[0], 'rb') as f:
            latest_model = pickle.load(f)
        if "metrics" not in latest_model:
            latest_model["metrics"] = results
            with open(model_files[0], 'wb') as f:
                pickle.dump(latest_model, f)
    
    return {
        "status": "success",
        "metrics": results,
        "timestamp": timestamp,
        "data_points": len(target_data)
    }

@app.get("/api/anomaly")
async def detect_anomalies():
    try:
        if df is None or not trained_models:
            raise HTTPException(status_code=400, detail="Dataset or models not available")
        
        # Find best model based on lowest RMSE
        best_model = min(model_metrics.keys(), key=lambda k: model_metrics[k]['RMSE'])
        predictions = trained_models[best_model]['predictions']
        actual = df['Magnitude'].values
        
        # Ensure arrays have the same length
        min_len = min(len(predictions), len(actual))
        predictions = predictions[:min_len]
        actual = actual[:min_len]
        
        if len(predictions) == 0 or len(actual) == 0:
            raise HTTPException(status_code=400, detail="No valid predictions or data available")
        
        residuals = actual - predictions
        
        # Handle cases where std is zero or very small
        residual_std = np.std(residuals)
        if residual_std == 0 or np.isnan(residual_std):
            # If no variation in residuals, no anomalies
            return {
                "anomaly_count": 0,
                "best_model": best_model,
                "threshold": 2.5,
                "anomalies": [],
                "total_points": len(actual)
            }
        
        # Z-score based anomaly detection
        z_scores = np.abs((residuals - np.mean(residuals)) / residual_std)
        threshold = 2.5
        anomaly_mask = z_scores > threshold
        anomaly_count = int(np.sum(anomaly_mask))
        
        # Get anomaly details
        anomaly_indices = np.where(anomaly_mask)[0]
        anomaly_data = []
        for idx in anomaly_indices[:10]:  # Limit to first 10 anomalies
            try:
                # Safely format datetime
                datetime_str = "N/A"
                if 'Datetime' in df.columns and idx < len(df):
                    dt_val = df.iloc[idx]['Datetime']
                    if pd.notna(dt_val):
                        datetime_str = dt_val.strftime('%Y-%m-%d %H:%M:%S')
                
                anomaly = {
                    "index": int(idx),
                    "datetime": datetime_str,
                    "actual": float(actual[idx]) if not (np.isnan(actual[idx]) or np.isinf(actual[idx])) else None,
                    "predicted": float(predictions[idx]) if not (np.isnan(predictions[idx]) or np.isinf(predictions[idx])) else None,
                    "residual": float(residuals[idx]) if not (np.isnan(residuals[idx]) or np.isinf(residuals[idx])) else None,
                    "z_score": float(z_scores[idx]) if not (np.isnan(z_scores[idx]) or np.isinf(z_scores[idx])) else None
                }
                anomaly_data.append(anomaly)
            except Exception as e:
                # Skip this anomaly if there's an error processing it
                print(f"Error processing anomaly at index {idx}: {e}")
                continue
        
        return {
            "anomaly_count": anomaly_count,
            "best_model": best_model,
            "threshold": threshold,
            "anomalies": anomaly_data,
            "total_points": len(actual)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting anomalies: {str(e)}")

@app.get("/api/anomalies")
async def detect_anomalies_plural():
    """Alias for /api/anomaly endpoint to handle plural requests"""
    return await detect_anomalies()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
