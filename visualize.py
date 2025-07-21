import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_training_history(history: dict, ticker: str) -> str:
    """
    Plot training and validation loss over epochs.
    
    Args:
        history: Training history from model.fit()
        ticker: Stock ticker symbol for the plot title
        
    Returns:
        Path to the saved plot
    """
    os.makedirs('visuals', exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    
    # Plot training & validation loss values
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    
    plt.title(f'Model Loss - {ticker}')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Save the plot
    plot_path = f'visuals/{ticker}_training_history.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_predictions(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    dates: np.ndarray, 
    ticker: str,
    future_dates: Optional[np.ndarray] = None,
    future_pred: Optional[np.ndarray] = None
) -> str:
    """
    Plot actual vs predicted stock prices.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        dates: Dates for the x-axis
        ticker: Stock ticker symbol
        future_dates: Dates for future predictions (optional)
        future_pred: Future predicted values (optional)
        
    Returns:
        Path to the saved plot
    """
    os.makedirs('visuals', exist_ok=True)
    
    plt.figure(figsize=(14, 7))
    
    # Plot actual values
    plt.plot(dates, y_true, label='Actual', color='blue', alpha=0.7)
    
    # Plot predicted values
    plt.plot(dates[-len(y_pred):], y_pred, label='Predicted', color='red', linestyle='--', alpha=0.9)
    
    # Plot future predictions if provided
    if future_pred is not None and future_dates is not None:
        plt.plot(future_dates, future_pred, label='Future Prediction', color='green', linestyle='-', marker='o')
    
    plt.title(f'Stock Price Prediction - {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_path = f'visuals/{ticker}_predictions.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_metrics(metrics: dict, ticker: str) -> str:
    """
    Plot model evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics to plot
        ticker: Stock ticker symbol
        
    Returns:
        Path to the saved plot
    """
    os.makedirs('visuals', exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    
    # Create bar plot of metrics
    names = list(metrics.keys())
    values = list(metrics.values())
    
    bars = plt.bar(names, values, color=['blue', 'orange', 'green'])
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
    
    plt.title(f'Model Performance Metrics - {ticker}')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_path = f'visuals/{ticker}_metrics.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_future_predictions(
    historical_dates: np.ndarray,
    historical_prices: np.ndarray,
    future_dates: np.ndarray,
    future_predictions: np.ndarray,
    ticker: str
) -> str:
    """
    Plot historical data with future predictions.
    
    Args:
        historical_dates: Dates for historical data
        historical_prices: Historical price data
        future_dates: Dates for future predictions
        future_predictions: Predicted future prices
        ticker: Stock ticker symbol
        
    Returns:
        Path to the saved plot
    """
    os.makedirs('visuals', exist_ok=True)
    
    plt.figure(figsize=(14, 7))
    
    # Plot historical data
    plt.plot(historical_dates, historical_prices, label='Historical', color='blue')
    
    # Plot the connecting point between historical and future
    if len(historical_dates) > 0 and len(future_dates) > 0:
        plt.plot([historical_dates[-1], future_dates[0]], 
                 [historical_prices[-1], future_predictions[0]], 
                 color='green', linestyle='--')
    
    # Plot future predictions
    plt.plot(future_dates, future_predictions, label='Future Prediction', 
             color='green', marker='o')
    
    plt.title(f'Future Price Prediction - {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_path = f'visuals/{ticker}_future_predictions.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    return plot_path
