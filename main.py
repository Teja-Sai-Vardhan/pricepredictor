#!/usr/bin/env python3
"""
Stock Price Predictor - Command Line Interface

This script provides a command-line interface for the Stock Price Predictor.
It allows users to train models and make predictions from the terminal.
"""

import argparse
import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import our utility functions
from data_utils import load_or_fetch_data, preprocess_data, train_test_split, DEFAULT_TICKERS
from model_utils import build_lstm_model, train_model, predict_future, cross_validate
from visualize import plot_training_history, plot_predictions, plot_metrics, plot_future_predictions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stock_predictor.log')
    ]
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Stock Price Predictor using LSTM')
    
    # Required arguments
    parser.add_argument('--ticker', type=str, default='AAPL',
                       help=f'Stock ticker symbol (default: AAPL). Available: {DEFAULT_TICKERS}')
    
    # Optional arguments
    parser.add_argument('--years', type=int, default=3,
                       help='Number of years of historical data to fetch (default: 3)')
    parser.add_argument('--forecast-days', type=int, default=20,
                       help='Number of days to forecast (default: 20)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--seq-length', type=int, default=60,
                       help='Sequence length for LSTM (default: 60)')
    parser.add_argument('--force-refetch', action='store_true',
                       help='Force refetch data instead of using cache')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable generating plots')
    
    return parser.parse_args()

def main():
    """Main function to run the stock price predictor."""
    args = parse_arguments()
    ticker = args.ticker.upper()
    
    logging.info(f"Starting Stock Price Predictor for {ticker}")
    logging.info(f"Parameters: {args}")
    
    try:
        # Step 1: Fetch data
        logging.info(f"Step 1/5: Fetching {ticker} data for the last {args.years} years...")
        df, status = load_or_fetch_data(ticker, args.years)
        
        if df is None:
            logging.error(f"Failed to fetch data: {status}")
            return 1
            
        logging.info(f"Successfully loaded data with {len(df)} records")
        
        # Step 2: Preprocess data
        logging.info("Step 2/5: Preprocessing data...")
        X, y, scaler = preprocess_data(df, seq_length=args.seq_length)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        
        logging.info(f"Data split - Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Step 3: Train model
        logging.info("Step 3/5: Training LSTM model...")
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        # Define model path
        model_path = f'models/{ticker}_lstm.h5'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1
        )
        
        # Save the model
        model.save(model_path)
        logging.info(f"Model saved to {model_path}")
        
        # Step 4: Evaluate model
        logging.info("Step 4/5: Evaluating model...")
        y_pred = model.predict(X_test, verbose=0).flatten()
        
        # Inverse transform the scaled data
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        r2 = r2_score(y_test_actual, y_pred_actual)
        
        logging.info(f"Evaluation Metrics:")
        logging.info(f"  MSE: {mse:.6f}")
        logging.info(f"  MAE: {mae:.6f}")
        logging.info(f"  R²: {r2:.6f}")
        
        # Step 5: Make future predictions
        logging.info("Step 5/5: Making future predictions...")
        last_sequence = X_test[-1:]
        future_predictions_scaled = predict_future(model, last_sequence, days=args.forecast_days)
        future_predictions = scaler.inverse_transform(future_predictions_scaled.reshape(-1, 1)).flatten()
        
        # Generate future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=args.forecast_days, freq='B')
        
        # Display predictions
        print("\n=== Future Price Predictions ===")
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_predictions
        })
        future_df['Change'] = future_df['Predicted_Price'].pct_change() * 100
        future_df['Cumulative_Change'] = (future_df['Predicted_Price'] / future_df['Predicted_Price'].iloc[0] - 1) * 100
        
        # Format the DataFrame for display
        pd.set_option('display.float_format', '{:.2f}'.format)
        print(future_df[['Date', 'Predicted_Price', 'Change', 'Cumulative_Change']].to_string(index=False))
        
        # Generate plots if not disabled
        if not args.no_plots:
            logging.info("Generating plots...")
            os.makedirs('visuals', exist_ok=True)
            
            # Plot training history
            plot_training_history(history.history, ticker)
            
            # Plot predictions vs actual
            plot_predictions(
                y_test_actual, y_pred_actual, 
                df.index[-len(y_test_actual):], ticker
            )
            
            # Plot future predictions
            plot_future_predictions(
                df.index[-60:],  # Last 60 days for context
                df['Close'].values[-60:],
                future_dates,
                future_predictions,
                ticker
            )
            
            # Plot metrics
            metrics = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2
            }
            plot_metrics(metrics, ticker)
            
            logging.info(f"Plots saved to the 'visuals' directory")
        
        logging.info("Analysis completed successfully!")
        return 0
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
