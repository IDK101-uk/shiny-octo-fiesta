"""
Trainer module for training neural network models.
"""

import os
import sys
import numpy as np
import logging
from datetime import datetime
import time
import joblib
from sklearn.model_selection import train_test_split

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_CONFIG, PATHS
from data.data_loader import DataLoader
from models.model1_bayesian_neural_network import BayesianNeuralNetwork

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class for training neural network models."""
    
    def __init__(self):
        """Initialize the model trainer."""
        self.data_loader = DataLoader()
        self.accuracy_threshold = MODEL_CONFIG["accuracy_threshold"]
        self.sequence_length = 60  # Number of time steps to use as input
        self.models = {}
        self.training_stats = {}
    
    def initialize_models(self, input_shape):
        """
        Initialize all the neural network models.
        
        Args:
            input_shape: Shape of the input data (sequence_length, num_features)
        """
        # Initialize all models with the same input shape
        self.models = {
            "model1": BayesianNeuralNetwork(input_shape),
            # Other models will be initialized here as they are implemented
        }
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def train_model(self, model_name, max_attempts=3, X=None, y=None):
        """
        Train a specific model until it reaches the accuracy threshold.
        
        Args:
            model_name: Name of the model to train
            max_attempts: Maximum number of training attempts
            X: Optional pre-loaded training data (features)
            y: Optional pre-loaded target data
            
        Returns:
            trained_model: Trained model if successful, None otherwise
            is_successful: Boolean indicating if training was successful
        """
        model = self.models.get(model_name)
        if model is None:
            logger.error(f"Model {model_name} not found")
            return None, False
        
        logger.info(f"Training {model_name}...")
        
        # Track training attempts
        attempts = 0
        is_successful = False
        training_stats = []
        
        # Set TensorFlow to log device placement
        import tensorflow as tf
        if MODEL_CONFIG.get("log_device_placement", False):
            tf.debugging.set_log_device_placement(True)
        else:
            tf.debugging.set_log_device_placement(False)
        
        # Function to clear memory
        def clear_memory():
            import gc
            import tensorflow as tf
            gc.collect()
            tf.keras.backend.clear_session()
            logger.info("Cleared memory and TensorFlow session")
        
        while attempts < max_attempts and not is_successful:
            clear_memory()  # Clear memory before each attempt
            attempts += 1
            logger.info(f"Training attempt {attempts}/{max_attempts} for {model_name}")
            
            try:
                # Use provided data if available, otherwise get new data
                if X is not None and y is not None:
                    logger.info(f"Using provided data for training. Shape: {X.shape}")
                    X_data, y_data = X, y
                else:
                    # Get a random month of data for training
                    logger.info("Loading new random month data for training")
                    success, message, data = self.data_loader.select_random_month()
                    if not success or data is None:
                        logger.error(f"Failed to load training data: {message}")
                        continue
                    
                    processed_data = self.data_loader.generate_features(data)
                    X_data, y_data = self.data_loader.prepare_training_data(processed_data, self.sequence_length)
                
                # Use a smaller subset of data if it's too large
                max_samples = 1000  # Limit to 1000 samples
                if len(X_data) > max_samples:
                    logger.info(f"Reducing training data from {len(X_data)} to {max_samples} samples")
                    X_data = X_data[:max_samples]
                    y_data = y_data[:max_samples]
                
                # Split data into training, validation, and test sets
                X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
                
                logger.info(f"Training data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
                logger.info(f"Validation data shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")
                logger.info(f"Test data shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")
                
                # Train the model
                start_time = time.time()
                logger.info(f"Starting model training for {model_name}...")
                try:
                    # Implement batch processing for large datasets
                    max_batch_size = MODEL_CONFIG.get("batch_size", 32)
                    train_size = len(X_train)
                    
                    logger.info(f"Dataset size: {train_size} samples")
                    
                    # Check if we need to use data chunking (extreme memory constraints)
                    use_data_chunking = train_size > 10000 and len(X_train[0]) > 40  # Large dataset with many features
                    
                    if use_data_chunking:
                        logger.info(f"Using data chunking strategy for very large dataset ({train_size} samples)")
                        
                        # Split data into chunks
                        chunk_size = 2000  # Process 2000 samples at a time
                        num_chunks = (train_size + chunk_size - 1) // chunk_size
                        logger.info(f"Splitting data into {num_chunks} chunks of {chunk_size} samples")
                        
                        # Train on each chunk incrementally
                        for chunk_idx in range(num_chunks):
                            start_idx = chunk_idx * chunk_size
                            end_idx = min(start_idx + chunk_size, train_size)
                            logger.info(f"Training on chunk {chunk_idx+1}/{num_chunks} (samples {start_idx}-{end_idx})")
                            
                            # Get chunk data
                            X_chunk = X_train[start_idx:end_idx]
                            y_chunk = y_train[start_idx:end_idx]
                            
                            # Train on chunk
                            if chunk_idx == 0:
                                # First chunk - normal training
                                history_chunk = model.train(X_chunk, y_chunk, X_val, y_val)
                            else:
                                # Subsequent chunks - continue training from current weights
                                model.model.fit(
                                    X_chunk, y_chunk,
                                    validation_data=(X_val, y_val),
                                    epochs=3,  # Fewer epochs for incremental training
                                    batch_size=model.batch_size,
                                    verbose=1
                                )
                            
                            # Clear memory after each chunk
                            clear_memory()
                        
                        # Final evaluation after all chunks
                        history = type('History', (), {})()
                        history.history = {'loss': [0], 'val_loss': [0]}  # Placeholder history
                    
                    # If dataset is large but not needing chunking, use gradient accumulation
                    elif train_size > 10000:
                        logger.info(f"Dataset is large ({train_size} samples). Using batch processing.")
                        
                        # Start with a reasonable batch size but allow for adjustment
                        try_batch_sizes = [max_batch_size, max_batch_size // 2, max_batch_size // 4, max_batch_size // 8, 1]
                        
                        for batch_size in try_batch_sizes:
                            try:
                                model.batch_size = batch_size
                                logger.info(f"Attempting training with batch size: {batch_size}")
                                history = model.train(X_train, y_train, X_val, y_val)
                                logger.info(f"Successfully trained with batch size: {batch_size}")
                                break
                            except tf.errors.ResourceExhaustedError:
                                logger.warning(f"Memory error with batch size {batch_size}, trying smaller batch size")
                                clear_memory()
                                # If we've tried the smallest batch size and still failed
                                if batch_size == 1:
                                    raise ValueError("Unable to train even with batch size of 1. Switching to data chunking strategy.")
                    else:
                        # For smaller datasets, use default approach with fallback
                        try:
                            history = model.train(X_train, y_train, X_val, y_val)
                        except tf.errors.ResourceExhaustedError:
                            logger.warning("Memory error with default batch size, trying with reduced batch size")
                            model.batch_size = max(1, model.batch_size // 2)
                            logger.info(f"Reduced batch size to {model.batch_size}")
                            history = model.train(X_train, y_train, X_val, y_val)
                    
                    training_time = time.time() - start_time
                    logger.info(f"Model training completed in {training_time:.2f} seconds")
                except Exception as train_error:
                    logger.error(f"Error during model.train(): {str(train_error)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
                
                # Evaluate the model
                logger.info(f"Evaluating model {model_name}...")
                try:
                    metrics = model.evaluate(X_test, y_test)
                except Exception as eval_error:
                    logger.error(f"Error during model evaluation: {str(eval_error)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
                
                # Check if the model reached the accuracy threshold
                if metrics["directional_accuracy"] >= self.accuracy_threshold:
                    is_successful = True
                    logger.info(f"Model {model_name} reached the accuracy threshold: {metrics['directional_accuracy']:.2%}")
                    
                    # Save the model
                    try:
                        model_path = model.save()
                        # Save the scalers used for this model
                        self.data_loader.save_scalers(model_name)
                    except Exception as save_error:
                        logger.error(f"Error saving model or scalers: {str(save_error)}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    # Plot training history and predictions
                    try:
                        model.plot_training_history(history)
                        model.plot_predictions(X_test, y_test)
                    except Exception as plot_error:
                        logger.error(f"Error creating plots: {str(plot_error)}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                else:
                    logger.info(f"Model {model_name} did not reach the accuracy threshold: {metrics['directional_accuracy']:.2%}")
                    
                    # If this was the last attempt, save the best model anyway
                    if attempts == max_attempts:
                        logger.warning(f"Max attempts reached for {model_name}. Saving the best model.")
                        try:
                            model_path = model.save()
                            self.data_loader.save_scalers(model_name)
                            model.plot_training_history(history)
                            model.plot_predictions(X_test, y_test)
                        except Exception as save_error:
                            logger.error(f"Error saving final model: {str(save_error)}")
                            import traceback
                            logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Record training stats
                train_stats = {
                    "attempt": attempts,
                    "training_time": training_time,
                    "epochs": len(history.history["loss"]),
                    "final_loss": history.history["loss"][-1],
                    "final_val_loss": history.history["val_loss"][-1],
                    "directional_accuracy": metrics["directional_accuracy"],
                    "mse": metrics["mse"],
                    "mae": metrics["mae"],
                    "is_successful": is_successful
                }
                training_stats.append(train_stats)
                
            except tf.errors.ResourceExhaustedError as mem_error:
                logger.error(f"Memory error during training: {str(mem_error)}")
                logger.warning("Reducing model complexity and batch size for next attempt")
                
                # Attempt to recover by clearing memory
                clear_memory()
                continue
                
            except Exception as e:
                logger.error(f"Unexpected error during training attempt {attempts}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        # Save training stats
        self.training_stats[model_name] = training_stats
        self._save_training_stats()
        
        return model, is_successful
    
    def train_all_models(self, X=None, y=None):
        """
        Train all models until they reach the accuracy threshold.
        
        Args:
            X: Optional pre-loaded training data (features)
            y: Optional pre-loaded target data
            
        Returns:
            successful_models: Dictionary of successfully trained models
        """
        successful_models = {}
        
        for model_name in self.models:
            model, is_successful = self.train_model(model_name, X=X, y=y)
            if is_successful:
                successful_models[model_name] = model
        
        logger.info(f"Successfully trained {len(successful_models)}/{len(self.models)} models")
        return successful_models
    
    def _save_training_stats(self):
        """Save training statistics to disk."""
        # Create stats directory if it doesn't exist
        stats_dir = os.path.join(PATHS["results_dir"], "training_stats")
        os.makedirs(stats_dir, exist_ok=True)
        
        # Save training stats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = os.path.join(stats_dir, f"training_stats_{timestamp}.joblib")
        joblib.dump(self.training_stats, stats_file)
        
        logger.info(f"Saved training statistics to {stats_file}")


def main():
    """Main function to run the model training."""
    # Create data loader and download latest data
    data_loader = DataLoader()
    data = data_loader.download_latest_data()
    
    # Process the data
    processed_data = data_loader.generate_features(data)
    
    # Prepare sample data to get input shape
    X_sample, _ = data_loader.prepare_training_data(processed_data)
    input_shape = X_sample.shape[1:]
    
    # Initialize trainer and models
    trainer = ModelTrainer()
    trainer.initialize_models(input_shape)
    
    # Train all models
    successful_models = trainer.train_all_models()
    
    return successful_models


if __name__ == "__main__":
    main()
