"""
Predictor module for making stock price predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import joblib
from datetime import datetime, timedelta
import tensorflow as tf
import matplotlib.pyplot as plt
import glob

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_CONFIG, PATHS, MODEL_CONFIG
from data.data_loader import DataLoader
from models.model1_bayesian_neural_network import BayesianNeuralNetwork

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockPredictor:
    """Class for making stock price predictions."""
    
    def __init__(self):
        """Initialize the stock predictor."""
        self.data_loader = DataLoader()
        self.models = {}
        self.sequence_length = 60  # Number of time steps to use as input
        self.prediction_horizon = MODEL_CONFIG["prediction_horizon"]
        
        # Create results directory if it doesn't exist
        os.makedirs(PATHS["results_dir"], exist_ok=True)
    
    def load_models(self):
        """
        Load all trained models.
        
        Returns:
            Dictionary of loaded models
        """
        model_dir = PATHS["model_dir"]
        loaded_models = {}
        
        # Find all model directories
        model_dirs = [d for d in glob.glob(f"{model_dir}/*") if os.path.isdir(d)]
        
        for model_dir in model_dirs:
            # Get model name from directory (e.g., BayesianNN_20230515_123456)
            model_name = os.path.basename(model_dir).split('_')[0]
            
            # Find corresponding metadata file
            metadata_files = glob.glob(f"{PATHS['model_dir']}/{os.path.basename(model_dir)}_metadata.joblib")
            if metadata_files:
                metadata = joblib.load(metadata_files[0])
                
                # Create model instance based on model name
                if model_name == "BayesianNN":
                    model = BayesianNeuralNetwork(metadata["input_shape"])
                    # Add more model types here as they are implemented
                else:
                    logger.warning(f"Unknown model type: {model_name}")
                    continue
                
                # Load model weights
                model.load(model_dir)
                
                # Load scalers
                try:
                    self.data_loader.load_scalers(model_name)
                except Exception as e:
                    logger.warning(f"Could not load scalers for {model_name}: {str(e)}")
                
                loaded_models[model_name] = model
                logger.info(f"Loaded model {model_name}")
        
        self.models = loaded_models
        return loaded_models
    
    def get_latest_data(self):
        """
        Get and process the latest data for prediction.
        
        Returns:
            Processed data ready for prediction
        """
        # Download the latest data
        data = self.data_loader.download_latest_data()
        if data is None:
            logger.error("Failed to download latest data")
            return None
        
        # Process the data
        processed_data = self.data_loader.generate_features(data)
        return processed_data
    
    def prepare_prediction_data(self, data):
        """
        Prepare the latest data for prediction.
        
        Args:
            data: Processed data
            
        Returns:
            X: Input data for prediction
        """
        # Select the last sequence_length data points
        if len(data) < self.sequence_length:
            logger.error(f"Not enough data points for prediction. Need at least {self.sequence_length}, got {len(data)}")
            return None
        
        # Select features and scale
        df = data[self.data_loader.features].copy()
        
        # Scale the data using the same scalers used during training
        if not self.data_loader.scalers:
            logger.error("Scalers not loaded. Call load_models() first.")
            return None
        
        scaled_data = pd.DataFrame()
        for column in df.columns:
            if column in self.data_loader.scalers:
                scaled_data[column] = self.data_loader.scalers[column].transform(df[column].values.reshape(-1, 1)).flatten()
            else:
                logger.warning(f"No scaler found for {column}, using raw values")
                scaled_data[column] = df[column]
        
        # Select the last sequence_length data points
        X = scaled_data.iloc[-self.sequence_length:].values.reshape(1, self.sequence_length, -1)
        
        return X
    
    def predict_next_candles(self, model_name=None):
        """
        Predict the next 5 candles.
        
        Args:
            model_name: Name of the model to use for prediction (default: use all models)
            
        Returns:
            Dictionary of predictions from each model
        """
        # Get the latest data
        data = self.get_latest_data()
        if data is None:
            return None
        
        # Prepare data for prediction
        X = self.prepare_prediction_data(data)
        if X is None:
            return None
        
        # Make predictions with all models or specific model
        predictions = {}
        if model_name is not None:
            if model_name in self.models:
                model = self.models[model_name]
                predictions[model_name] = self._make_prediction(model, X, data)
            else:
                logger.error(f"Model {model_name} not found")
        else:
            for name, model in self.models.items():
                predictions[name] = self._make_prediction(model, X, data)
        
        # Save predictions
        self._save_predictions(predictions, data)
        
        return predictions
    
    def _make_prediction(self, model, X, data):
        """
        Make prediction with a specific model.
        
        Args:
            model: Model to use for prediction
            X: Input data
            data: Original data for reference
            
        Returns:
            Dictionary with prediction details
        """
        try:
            # Make prediction
            if hasattr(model, 'predict_with_uncertainty'):
                # For models with uncertainty estimation
                y_pred, y_std = model.predict_with_uncertainty(X)
                lower_bound, upper_bound = model.get_prediction_intervals(X)
                
                # Inverse transform the predictions
                close_scaler = self.data_loader.scalers["Close"]
                y_pred_orig = close_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                lower_bound_orig = close_scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
                upper_bound_orig = close_scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten()
                
                prediction = {
                    "predicted_values": y_pred_orig.tolist(),
                    "lower_bound": lower_bound_orig.tolist(),
                    "upper_bound": upper_bound_orig.tolist(),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                # For regular models
                y_pred = model.predict(X)
                
                # Inverse transform the predictions
                close_scaler = self.data_loader.scalers["Close"]
                y_pred_orig = close_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                
                prediction = {
                    "predicted_values": y_pred_orig.tolist(),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            
            # Calculate time points
            last_time = data.index[-1]
            time_delta = pd.Timedelta(minutes=5)  # 5-minute candles
            future_times = [last_time + (i+1)*time_delta for i in range(self.prediction_horizon)]
            prediction["time_points"] = [t.strftime("%Y-%m-%d %H:%M:%S") for t in future_times]
            
            return prediction
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def _save_predictions(self, predictions, data):
        """
        Save predictions to disk.
        
        Args:
            predictions: Dictionary of predictions
            data: Original data for reference
        """
        # Create predictions directory if it doesn't exist
        predictions_dir = os.path.join(PATHS["results_dir"], "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Save predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_file = os.path.join(predictions_dir, f"predictions_{timestamp}.json")
        
        # Add last known prices for reference
        last_candle = data.iloc[-1].to_dict()
        last_prices = {
            "time": data.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(last_candle["Open"]),
            "high": float(last_candle["High"]),
            "low": float(last_candle["Low"]),
            "close": float(last_candle["Close"]),
            "volume": float(last_candle["Volume"])
        }
        
        # Create combined predictions data
        predictions_data = {
            "timestamp": timestamp,
            "last_candle": last_prices,
            "predictions": predictions
        }
        
        # Save as JSON
        import json
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=4)
        
        logger.info(f"Saved predictions to {predictions_file}")
        
        # Plot predictions
        self._plot_predictions(predictions, data)
    
    def _plot_predictions(self, predictions, data):
        """
        Plot predictions.
        
        Args:
            predictions: Dictionary of predictions
            data: Original data for reference
        """
        plt.figure(figsize=(12, 8))
        
        # Plot historical data
        historical_prices = data["Close"].values[-20:]  # Last 20 points
        historical_times = list(range(len(historical_prices)))
        plt.plot(historical_times, historical_prices, 'b-', label='Historical')
        
        # Plot predictions for each model
        colors = ['r', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'lime', 'brown']
        next_point = len(historical_prices)
        for i, (model_name, prediction) in enumerate(predictions.items()):
            if prediction is not None:
                pred_values = prediction["predicted_values"]
                pred_times = list(range(next_point, next_point + len(pred_values)))
                color = colors[i % len(colors)]
                plt.plot(pred_times, pred_values, f'{color}-', label=f'{model_name}')
                
                # Plot prediction intervals if available
                if "lower_bound" in prediction and "upper_bound" in prediction:
                    plt.fill_between(
                        pred_times,
                        prediction["lower_bound"],
                        prediction["upper_bound"],
                        color=color,
                        alpha=0.2
                    )
        
        # Add a vertical line at the current time
        plt.axvline(x=next_point-1, color='k', linestyle='--')
        
        # Add labels and title
        plt.xlabel('Time (5-minute intervals)')
        plt.ylabel('SPY Price')
        plt.title('SPY Price Prediction')
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        os.makedirs(PATHS["results_dir"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(PATHS["results_dir"], f"prediction_plot_{timestamp}.png")
        plt.savefig(fig_path)
        plt.close()
        
        logger.info(f"Prediction plot saved to {fig_path}")


def main():
    """Main function to run the predictor."""
    # Initialize predictor
    predictor = StockPredictor()
    
    # Load models
    predictor.load_models()
    
    # Make predictions
    predictions = predictor.predict_next_candles()
    
    return predictions


if __name__ == "__main__":
    main()
