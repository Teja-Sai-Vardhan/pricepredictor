import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default tickers for the application
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'WMT']

def fetch_stock_data(ticker: str, years: int = 3, retries: int = 3) -> Optional[pd.DataFrame]:
    """
    Fetch historical stock data using yfinance.
    
    Args:
        ticker: Stock ticker symbol
        years: Number of years of historical data to fetch
        retries: Number of retry attempts
        
    Returns:
        DataFrame with stock data or None if fetch fails
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    
    for attempt in range(retries):
        try:
            logging.info(f"Fetching {ticker} data (attempt {attempt + 1}/{retries})")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                logging.warning(f"No data found for {ticker}")
                return None
                
            # Keep only the 'Close' column for univariate analysis
            df = df[['Close']].copy()
            df.columns = ['Close']
            
            # Save raw data
            os.makedirs('data', exist_ok=True)
            df.to_csv(f'data/raw_{ticker}.csv')
            
            return df
            
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == retries - 1:
                logging.error(f"Failed to fetch data for {ticker} after {retries} attempts")
                return None
            
            # Exponential backoff
            time.sleep(2 ** attempt)

def preprocess_data(df: pd.DataFrame, seq_length: int = 60) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Preprocess the data for LSTM training.
    
    Args:
        df: DataFrame with 'Close' prices
        seq_length: Length of sequences for LSTM
        
    Returns:
        Tuple of (X, y, scaler) for training
    """
    # Create a copy to avoid SettingWithCopyWarning
    data = df[['Close']].copy()
    
    # Handle missing values
    data = data.fillna(method='ffill')
    data = data.dropna()
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:(i + seq_length), 0])
        y.append(scaled_data[i + seq_length, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> tuple:
    """
    Split data into training and testing sets.
    
    Args:
        X: Input features
        y: Target values
        test_size: Proportion of data to use for testing
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    train_size = int(len(X) * (1 - test_size))
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:]

def load_or_fetch_data(ticker: str, years: int = 3) -> Optional[Tuple[pd.DataFrame, str]]:
    """
    Load data from cache or fetch fresh data.
    
    Args:
        ticker: Stock ticker symbol
        years: Number of years of data to fetch
        
    Returns:
        Tuple of (DataFrame, status_message) or (None, error_message)
    """
    cache_file = f'data/raw_{ticker}.csv'
    
    try:
        # Try to load from cache
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df, f"Loaded {ticker} data from cache"
            
        # Fetch fresh data if not in cache
        df = fetch_stock_data(ticker, years)
        if df is not None:
            return df, f"Successfully fetched {ticker} data"
        else:
            return None, f"Failed to fetch data for {ticker}"
            
    except Exception as e:
        logging.error(f"Error in load_or_fetch_data: {str(e)}")
        return None, f"Error: {str(e)}"
