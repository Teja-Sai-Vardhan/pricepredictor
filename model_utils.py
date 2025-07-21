import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_lstm_model(input_shape: tuple) -> Sequential:
    """
    Build an LSTM model for time series prediction.
    
    Args:
        input_shape: Shape of input data (timesteps, features)
    """
    # Configure TensorFlow to use CPU only for compatibility
    tf.config.set_visible_devices([], 'GPU')
    
    # Build the model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    # Compile with standard Adam optimizer
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_model(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: Optional[np.ndarray] = None, 
    y_val: Optional[np.ndarray] = None,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
    model_path: str = 'models/lstm_model.keras'
) -> Tuple[Sequential, dict]:
    """
    Train the LSTM model with early stopping and model checkpointing.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        patience: Number of epochs to wait before early stopping
        model_path: Path to save the best model
        
    Returns:
        Trained model and training history
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Build the model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', 
                     patience=patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_path, save_best_only=True, 
                      monitor='val_loss' if X_val is not None else 'loss',
                      save_weights_only=False, mode='min', verbose=1)
    ]
    
    # Train the model with error handling
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load the best model weights
        model = tf.keras.models.load_model(model_path, compile=True)
        
        return model, history.history
        
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        # Try to save the model even if training was interrupted
        model.save(model_path)
        raise

def cross_validate(X: np.ndarray, y: np.ndarray, n_splits: int = 5, epochs: int = 50) -> dict:
    """
    Perform k-fold cross-validation on the LSTM model.
    
    Args:
        X: Input features
        y: Target values
        n_splits: Number of folds
        epochs: Number of epochs per fold
        
    Returns:
        Dictionary with cross-validation results
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        logging.info(f"Training fold {fold + 1}/{n_splits}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate
        score = model.evaluate(X_val, y_val, verbose=0)
        cv_scores.append(score)
        logging.info(f"Fold {fold + 1} - MSE: {score:.6f}")
    
    return {
        'mean_mse': np.mean(cv_scores),
        'std_mse': np.std(cv_scores),
        'all_scores': cv_scores
    }

def predict_future(model: Sequential, last_sequence: np.ndarray, days: int = 20) -> np.ndarray:
    """
    Predict future stock prices using the trained model.
    
    Args:
        model: Trained Keras model
        last_sequence: Last sequence of scaled data points (shape: [1, seq_length, 1])
        days: Number of days to predict
        
    Returns:
        Array of predicted values
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # Predict next value
        next_pred = model.predict(current_sequence, verbose=0)[0][0]
        predictions.append(next_pred)
        
        # Update sequence with the new prediction
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred
    
    return np.array(predictions)

def load_or_train_model(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    model_path: str = 'models/lstm_model.h5',
    force_retrain: bool = False
) -> Sequential:
    """
    Load a trained model or train a new one if not found.
    
    Args:
        X_train: Training features
        y_train: Training targets
        model_path: Path to the model file
        force_retrain: If True, train a new model even if one exists
        
    Returns:
        Trained Keras model
    """
    if not force_retrain and os.path.exists(model_path):
        logging.info(f"Loading model from {model_path}")
        return load_model(model_path)
    
    logging.info("Training new model...")
    model, _ = train_model(X_train, y_train, model_path=model_path)
    return model
