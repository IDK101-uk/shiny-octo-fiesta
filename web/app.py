"""
Flask web application for the SPY stock price prediction system.
"""

import os
import sys
import json
import logging
import pandas as pd
from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory
from datetime import datetime

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import WEB_CONFIG, PATHS
from data.polygon_manager import PolygonManager
from prediction.predictor import StockPredictor
from training.trainer import ModelTrainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add a file handler for error logging
os.makedirs(PATHS["logs_dir"], exist_ok=True)
error_log_file = os.path.join(PATHS["logs_dir"], 'training_errors.log')
file_handler = logging.FileHandler(error_log_file)
file_handler.setLevel(logging.ERROR)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Initialize global objects
# Using Polygon.io for SPY 1-minute data
data_loader = PolygonManager()
predictor = StockPredictor()
trainer = None  # Will be initialized when needed to save memory

# Global variables for tracking training status
training_status = {
    "in_progress": False,
    "model_name": None,
    "progress": 0,
    "status_message": "Not started",
    "last_updated": None
}

# Training background processes
training_processes = {}

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Render the dashboard page."""
    return render_template('dashboard.html', 
                          training_status=training_status)

@app.route('/api/data/download/latest', methods=['POST'])
def download_latest_data():
    """API endpoint to download the latest SPY 1-minute data"""
    try:
        # Force refresh if requested
        force_refresh = request.json.get('force_refresh', False)
        days = request.json.get('days', 7)
        
        # Download latest data
        success, message, data_or_stats = data_loader.download_latest_data(days=days, force_refresh=force_refresh)
        
        # Convert DataFrame to stats if necessary
        stats = {}
        if success and isinstance(data_or_stats, pd.DataFrame):
            df = data_or_stats
            stats = {
                "total_rows": len(df),
                "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                "trading_days": len(df['timestamp'].dt.date.unique()),
                "average_rows_per_day": int(len(df) / max(len(df['timestamp'].dt.date.unique()), 1))
            }
        elif success and isinstance(data_or_stats, dict):
            stats = data_or_stats
        
        return jsonify({
            'success': success,
            'message': message,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error in download_latest_data: {e}")
        return jsonify({
            'success': False,
            'message': f"Error: {str(e)}"
        })

@app.route('/api/data/download/historical', methods=['POST'])
def download_historical_data():
    """API endpoint to download historical SPY 1-minute data for training"""
    try:
        # Get number of years to download (default to 2)
        years = request.json.get('years', 2)
        
        # Download historical data
        success, message, data_or_stats = data_loader.download_historical_data(years=years)
        
        # Convert DataFrame to stats if necessary
        stats = {}
        if success and isinstance(data_or_stats, pd.DataFrame):
            df = data_or_stats
            # Calculate statistics from the data
            stats = {
                "total_rows": len(df),
                "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                "years_covered": len(df['timestamp'].dt.year.unique()),
                "months_covered": len(df['timestamp'].dt.strftime('%Y-%m').unique())
            }
        elif success and isinstance(data_or_stats, dict):
            stats = data_or_stats
        
        return jsonify({
            'success': success,
            'message': message,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error in download_historical_data: {e}")
        return jsonify({
            'success': False,
            'message': f"Error: {str(e)}"
        })

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """API endpoint to start model training."""
    global trainer, training_status
    
    try:
        # Check if training is already in progress
        if training_status["in_progress"]:
            return jsonify({
                "success": False, 
                "message": f"Training already in progress for {training_status['model_name']}"
            }), 400
        
        # Get request parameters
        data = request.get_json() or {}
        model_name = data.get("model_name")  # If None, will train all models
        
        # Update training status
        training_status = {
            "in_progress": True,
            "model_name": model_name if model_name else "all models",
            "progress": 0,
            "status_message": "Initializing training...",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Clean up memory before starting
        import gc
        import tensorflow as tf
        gc.collect()
        tf.keras.backend.clear_session()
        
        # Initialize trainer if not already done
        if trainer is None:
            logger.info("Initializing ModelTrainer")
            trainer = ModelTrainer()
        
        # Start training process
        # In a production environment, this would be a background task
        # For simplicity, we're running it synchronously here
        
        # Download and process data for training - random month from historical data
        training_status["status_message"] = "Selecting random month of historical data..."
        training_status["progress"] = 5
        success, message, data = data_loader.select_random_month(from_years=2)  # Explicitly set to 2 years
        
        if not success or data is None:
            training_status["in_progress"] = False
            training_status["status_message"] = f"Failed to load training data: {message}"
            return jsonify({"success": False, "message": message}), 500
        
        # Process the data
        training_status["status_message"] = "Generating features..."
        training_status["progress"] = 15
        try:
            processed_data = data_loader.generate_features(data)
            logger.info(f"Features generated successfully for {len(processed_data)} rows")
        except Exception as feature_error:
            logger.error(f"Error generating features: {str(feature_error)}")
            training_status["in_progress"] = False
            training_status["status_message"] = f"Error generating features: {str(feature_error)}"
            return jsonify({"success": False, "message": f"Error generating features: {str(feature_error)}"}), 500
        
        # Prepare sample data to get input shape
        training_status["status_message"] = "Preparing training data..."
        training_status["progress"] = 25
        try:
            X_sample, y_sample = data_loader.prepare_training_data(processed_data)
            input_shape = X_sample.shape[1:]
            logger.info(f"Prepared training data with shape {X_sample.shape}")
            logger.info(f"Target data shape: {y_sample.shape}")
            logger.info(f"Input shape for model: {input_shape}")
            
            if len(y_sample.shape) == 1:
                # Reshape y_sample to 2D if it's 1D (needed for the prediction_horizon)
                logger.info("Reshaping target data for model compatibility")
                y_sample = y_sample.reshape(-1, 1)
        except Exception as data_prep_error:
            logger.error(f"Error preparing training data: {str(data_prep_error)}")
            training_status["in_progress"] = False
            training_status["status_message"] = f"Error preparing training data: {str(data_prep_error)}"
            return jsonify({"success": False, "message": f"Error preparing training data: {str(data_prep_error)}"}), 500
        
        # Use the full dataset for training (remove the limit)
        logger.info(f"Using full dataset with {len(X_sample)} samples for training")
        
        # Initialize models
        training_status["status_message"] = "Initializing models..."
        training_status["progress"] = 30
        try:
            trainer.initialize_models(input_shape)
            logger.info(f"Models initialized with input shape {input_shape}")
        except Exception as model_init_error:
            logger.error(f"Error initializing models: {str(model_init_error)}")
            training_status["in_progress"] = False
            training_status["status_message"] = f"Error initializing models: {str(model_init_error)}"
            return jsonify({"success": False, "message": f"Error initializing models: {str(model_init_error)}"}), 500
        
        # Start training
        training_status["status_message"] = "Training models..."
        training_status["progress"] = 40
        
        try:
            import tensorflow as tf
            # Configure TensorFlow
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    logger.error(f"Error configuring GPU: {e}")
                    
            # Limit TensorFlow memory usage
            if not gpus:
                # For CPU, limit memory usage
                tf.config.threading.set_intra_op_parallelism_threads(1)
                tf.config.threading.set_inter_op_parallelism_threads(1)
            
            # Log input shapes and memory info
            logger.info(f"Training with data of shape X:{X_sample.shape}, y:{y_sample.shape}")
            import psutil
            mem = psutil.virtual_memory()
            logger.info(f"Memory before training: {mem.percent}% used, {mem.available / (1024**2):.2f}MB available")
            
            # Catch specific error types
            try:
                if model_name:
                    # Train specific model
                    logger.info(f"Starting training for model: {model_name}")
                    model, is_successful = trainer.train_model(model_name, X=X_sample, y=y_sample)
                    successful_models = {model_name: model} if is_successful else {}
                else:
                    # Train all models
                    logger.info("Starting training for all models")
                    successful_models = trainer.train_all_models(X=X_sample, y=y_sample)
                
                # Update status
                training_status["in_progress"] = False
                training_status["progress"] = 100
                training_status["status_message"] = f"Training completed successfully for {len(successful_models)}/{len(trainer.models)} models"
                training_status["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Clean up memory after training
                gc.collect()
                tf.keras.backend.clear_session()
                
                return jsonify({
                    "success": True, 
                    "message": f"Training completed for {len(successful_models)}/{len(trainer.models)} models",
                    "successful_models": list(successful_models.keys())
                })
            
            except tf.errors.ResourceExhaustedError as memory_error:
                logger.error(f"Out of memory error: {str(memory_error)}")
                error_message = "System ran out of memory while training. Try reducing the model complexity or batch size."
                
                # Clean up memory
                gc.collect()
                tf.keras.backend.clear_session()
                
                # Update status
                training_status["in_progress"] = False
                training_status["status_message"] = error_message
                training_status["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                return jsonify({"success": False, "message": error_message}), 500
                
            except tf.errors.InvalidArgumentError as arg_error:
                logger.error(f"TensorFlow invalid argument error: {str(arg_error)}")
                error_message = f"Invalid argument error in TensorFlow: {str(arg_error)}"
                
                # Update status
                training_status["in_progress"] = False
                training_status["status_message"] = error_message
                training_status["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                return jsonify({"success": False, "message": error_message}), 500
            
            except Exception as training_error:
                logger.error(f"Unexpected error during training: {str(training_error)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                error_message = f"Unexpected error during training: {str(training_error)}"
                
                # Update status
                training_status["in_progress"] = False
                training_status["status_message"] = error_message
                training_status["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                return jsonify({"success": False, "message": error_message}), 500
                
        except Exception as e:
            # This catches errors before the training even starts
            logger.error(f"Error preparing for training: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Clean up memory
            gc.collect()
            
            # Update status
            training_status["in_progress"] = False
            training_status["status_message"] = f"Error preparing for training: {str(e)}"
            training_status["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return jsonify({"success": False, "message": f"Error preparing for training: {str(e)}"}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Traceback: {error_trace}")
        
        # Create a more detailed error message for the UI
        error_message = f"Error: {str(e)}"
        if "shape" in error_trace.lower() or "dimension" in error_trace.lower():
            error_message = "Data shape error: Model input/output dimensions mismatch. Check logs for details."
        elif "memory" in error_trace.lower():
            error_message = "Memory error: Not enough memory to train the model. Try with smaller dataset."
        elif "cuda" in error_trace.lower() or "gpu" in error_trace.lower():
            error_message = "GPU error: Issue with GPU acceleration. Check CUDA installation."
        
        # Update status
        training_status["in_progress"] = False
        training_status["status_message"] = error_message
        training_status["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Clean up memory
        import gc
        gc.collect()
        
        return jsonify({"success": False, "message": error_message}), 500

@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    """API endpoint to get current training status."""
    return jsonify(training_status)

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """API endpoint to make predictions."""
    try:
        # Get request parameters
        data = request.get_json() or {}
        model_name = data.get("model_name")  # If None, will use all models
        
        # Load models if not already loaded
        if not predictor.models:
            predictor.load_models()
            if not predictor.models:
                return jsonify({"success": False, "message": "No trained models found"}), 404
        
        # Make predictions
        predictions = predictor.predict_next_candles(model_name)
        if predictions is None:
            return jsonify({"success": False, "message": "Failed to make predictions"}), 500
        
        # Get the latest prediction plot file
        import glob
        plot_files = sorted(glob.glob(os.path.join(PATHS["results_dir"], "prediction_plot_*.png")))
        latest_plot = plot_files[-1] if plot_files else None
        
        # Format response
        response = {
            "success": True,
            "predictions": predictions,
            "plot_url": f"/plots/{os.path.basename(latest_plot)}" if latest_plot else None
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """API endpoint to get information about available models."""
    try:
        # Load models if not already loaded
        if not predictor.models:
            predictor.load_models()
        
        # Get model information
        models_info = {}
        for name, model in predictor.models.items():
            models_info[name] = {
                "name": name,
                "type": model.__class__.__name__,
                "accuracy": predictor.model_accuracies.get(name, "Unknown")
            }
        
        return jsonify({
            "success": True,
            "models": models_info
        })
    
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@app.route('/api/plots', methods=['GET'])
def get_plots():
    """API endpoint to get available plots."""
    try:
        # Get all prediction plots
        import glob
        plot_files = sorted(glob.glob(os.path.join(PATHS["results_dir"], "prediction_plot_*.png")))
        
        # Format response
        plots = []
        for plot_file in plot_files:
            timestamp = os.path.basename(plot_file).replace("prediction_plot_", "").replace(".png", "")
            plots.append({
                "url": f"/plots/{os.path.basename(plot_file)}",
                "timestamp": timestamp
            })
        
        return jsonify({
            "success": True,
            "plots": plots
        })
    
    except Exception as e:
        logger.error(f"Error getting plots: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@app.route('/plots/<filename>')
def serve_plot(filename):
    """Serve plot images."""
    return send_from_directory(PATHS["results_dir"], filename)

def main():
    """Run the web application."""
    app.run(host=WEB_CONFIG["host"], port=WEB_CONFIG["port"], debug=WEB_CONFIG["debug"])

if __name__ == "__main__":
    main() 