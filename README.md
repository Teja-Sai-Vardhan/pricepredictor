# 📈 Stock Price Predictor

A powerful stock price prediction tool built with Streamlit and TensorFlow LSTM models. This application helps you forecast stock prices using historical data and machine learning.

## 🚀 Features

- **Interactive Web Interface**: User-friendly interface built with Streamlit
- **Multiple Stocks**: Supports various stock tickers (AAPL, MSFT, GOOGL, etc.)
- **Customizable Parameters**: Adjust model parameters for better predictions
- **Visual Analytics**: Beautiful visualizations of historical and predicted prices
- **Model Performance**: Detailed metrics and evaluation
- **Data Collection**: Fetches historical stock data using the Yahoo Finance API
- **Data Preprocessing**: Handles missing values, scales data, and creates sequences for LSTM
- **LSTM Model**: Implements a deep learning model for time series forecasting
- **Model Evaluation**: Provides metrics like MSE, MAE, and R² score
- **Visualization**: Generates interactive plots of predictions and model performance
- **Web Interface**: User-friendly Streamlit app for easy interaction
- **CLI Support**: Command-line interface for batch processing

## 🛠️ Installation
## Tech Stack

- **Python 3.10+**
- **TensorFlow 2.x** - For building and training the LSTM model
- **yfinance** - For fetching stock market data
- **pandas & NumPy** - For data manipulation and numerical operations
- **scikit-learn** - For data preprocessing and model evaluation
- **Matplotlib** - For data visualization
- **Streamlit** - For the web interface

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/stock-price-predictor.git
   cd stock-price-predictor
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv stock_env
   source stock_env/bin/activate  # On Windows: stock_env\Scripts\activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Interface

To launch the Streamlit web app:

```bash
streamlit run app.py
```

Then open your web browser to the URL shown in the terminal (usually http://localhost:8501).

### Command Line Interface

To use the command-line interface:

```bash
python main.py --ticker AAPL --years 3 --forecast-days 20 --epochs 50
```

#### Command Line Arguments

- `--ticker`: Stock ticker symbol (default: AAPL)
- `--years`: Number of years of historical data to fetch (default: 3)
- `--forecast-days`: Number of days to forecast (default: 20)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 32)
- `--seq-length`: Sequence length for LSTM (default: 60)
- `--force-refetch`: Force refetch data instead of using cache
- `--no-plots`: Disable generating plots

## Project Structure

```
stock-price-predictor/
├── data/                   # Directory for storing raw and processed data
├── models/                 # Directory for saving trained models
├── visuals/                # Directory for saving generated plots
├── app.py                 # Streamlit web application
├── data_utils.py          # Data fetching and preprocessing utilities
├── main.py                # Command-line interface
├── model_utils.py         # LSTM model building and training
├── requirements.txt       # Python dependencies
└── visualize.py           # Data visualization functions
```

## How It Works

1. **Data Collection**: The application fetches historical stock data using the Yahoo Finance API.
2. **Data Preprocessing**: The data is cleaned, scaled, and transformed into sequences suitable for LSTM training.
3. **Model Training**: An LSTM neural network is trained on the historical data to learn patterns in stock price movements.
4. **Prediction**: The trained model is used to forecast future stock prices.
5. **Visualization**: The results are visualized using Matplotlib, showing both historical and predicted prices.

## Model Architecture

The LSTM model consists of:

- Two LSTM layers with 50 units each
- Dropout layers (0.2 dropout rate) for regularization
- Dense layers for prediction
- Adam optimizer and mean squared error loss function

## Performance

The model's performance is evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

## Limitations

- The model is trained on historical data and may not account for unforeseen market events.
- Stock prices are influenced by numerous external factors that may not be captured by the model.
- Past performance is not indicative of future results.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Yahoo Finance for providing the stock market data
- TensorFlow and Keras for the deep learning framework
- The open-source community for various libraries used in this project

## Disclaimer

This project is for educational purposes only and should not be considered as financial advice. Always do your own research before making investment decisions.
