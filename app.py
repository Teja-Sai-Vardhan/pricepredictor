import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os
import time

# Import our utility functions
from data_utils import load_or_fetch_data, preprocess_data, train_test_split, DEFAULT_TICKERS
from model_utils import build_lstm_model, train_model, predict_future, load_or_train_model, cross_validate
from visualize import plot_training_history, plot_predictions, plot_metrics, plot_future_predictions

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Basic styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Stock Price Predictor")
st.write("Predict future stock prices using LSTM neural networks.")

# Sidebar for user inputs
st.sidebar.header("Stock Selection")

# Ticker selection
ticker = st.sidebar.selectbox(
    "Select a stock ticker:",
    DEFAULT_TICKERS,
    index=0
)

# Date range selection
years = st.sidebar.slider(
    "Years of historical data:",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

# Forecast days
forecast_days = st.sidebar.slider(
    "Days to forecast:",
    min_value=5,
    max_value=60,
    value=20,
    step=5
)

# Model parameters
st.sidebar.header("Model Parameters")
epochs = st.sidebar.slider(
    "Number of epochs:",
    min_value=10,
    max_value=200,
    value=50,
    step=10
)

batch_size = st.sidebar.selectbox(
    "Batch size:",
    [16, 32, 64, 128],
    index=1
)

# Main content
st.subheader(f"Analyzing: {ticker}")

# Add a button to trigger the analysis
if st.button("Run Analysis"):
    with st.spinner('Fetching data and training model...'):
        try:
            # Create a placeholder for the progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Fetch data
            status_text.text("Step 1/4: Fetching stock data...")
            progress_bar.progress(10)
            
            df, status = load_or_fetch_data(ticker, years)
            
            if df is None:
                st.error(f"Failed to fetch data: {status}")
                st.stop()
            
            # Step 2: Preprocess data
            status_text.text("Step 2/4: Preprocessing data...")
            progress_bar.progress(30)
            
            # Prepare data for LSTM
            X, y, scaler = preprocess_data(df)
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            
            # Step 3: Train model
            status_text.text("Step 3/4: Training LSTM model...")
            progress_bar.progress(60)
            
            # Build and train the model
            model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            
            # Step 4: Make predictions
            status_text.text("Step 4/4: Making predictions...")
            progress_bar.progress(80)
            
            # Predict on test set
            y_pred = model.predict(X_test, verbose=0).flatten()
            
            # Inverse transform the scaled data
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            mse = mean_squared_error(y_test_actual, y_pred_actual)
            mae = mean_absolute_error(y_test_actual, y_pred_actual)
            r2 = r2_score(y_test_actual, y_pred_actual)
            
            # Predict future prices
            last_sequence = X_test[-1:]
            future_predictions_scaled = predict_future(model, last_sequence, days=forecast_days)
            future_predictions = scaler.inverse_transform(future_predictions_scaled.reshape(-1, 1)).flatten()
            
            # Generate future dates
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='B')
            
            # Update progress
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            time.sleep(0.5)
            
            # Display results
            st.success("Analysis completed successfully!")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Historical Predictions", "Future Forecast", "Model Performance", "Metrics"])
            
            with tab1:
                st.subheader("Historical Price Predictions")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot actual prices
                ax.plot(df.index[-len(y_test):], y_test_actual, label='Actual', color='blue')
                
                # Plot predicted prices
                ax.plot(df.index[-len(y_test):], y_pred_actual, label='Predicted', color='red', linestyle='--')
                
                ax.set_title(f'{ticker} Stock Price Prediction')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab2:
                st.subheader(f"Next {forecast_days} Days Forecast")
                
                # Create a DataFrame for the forecast
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted_Price': future_predictions
                })
                forecast_df.set_index('Date', inplace=True)
                
                # Plot the forecast
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot historical data (last 60 days for context)
                historical_dates = df.index[-60:]
                historical_prices = df['Close'].values[-60:]
                ax.plot(historical_dates, historical_prices, label='Historical', color='blue')
                
                # Plot the connecting point
                if len(historical_dates) > 0 and len(future_dates) > 0:
                    ax.plot(
                        [historical_dates[-1], future_dates[0]],
                        [historical_prices[-1], future_predictions[0]],
                        color='green', linestyle='--'
                    )
                
                # Plot forecast
                ax.plot(future_dates, future_predictions, label='Forecast', color='green', marker='o')
                
                ax.set_title(f'{ticker} {forecast_days}-Day Price Forecast')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show forecast table
                st.subheader("Forecast Details")
                forecast_df['Change'] = forecast_df['Predicted_Price'].pct_change() * 100
                forecast_df['Cumulative_Change'] = (forecast_df['Predicted_Price'] / forecast_df['Predicted_Price'].iloc[0] - 1) * 100
                st.dataframe(forecast_df.style.format({
                    'Predicted_Price': '${:,.2f}',
                    'Change': '{:.2f}%',
                    'Cumulative_Change': '{:.2f}%'
                }))
            
            with tab3:
                st.subheader("Model Training History")
                
                # Plot training & validation loss
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(history.history['loss'], label='Training Loss')
                
                if 'val_loss' in history.history:
                    ax.plot(history.history['val_loss'], label='Validation Loss')
                
                ax.set_title('Model Loss')
                ax.set_ylabel('Loss (MSE)')
                ax.set_xlabel('Epoch')
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Model summary
                st.subheader("Model Architecture")
                from io import StringIO
                import sys
                
                # Capture model summary
                old_stdout = sys.stdout
                sys.stdout = buffer = StringIO()
                model.summary(print_fn=lambda x: buffer.write(x + '\n'))
                sys.stdout = old_stdout
                model_summary = buffer.getvalue()
                
                st.text(model_summary)
            
            with tab4:
                st.subheader("Performance Metrics")
                
                # Create metrics DataFrame
                metrics = {
                    'Metric': ['Mean Squared Error (MSE)', 'Mean Absolute Error (MAE)', 'R² Score'],
                    'Value': [mse, mae, r2]
                }
                
                # Display metrics in a table
                st.table(pd.DataFrame(metrics).style.format({
                    'Value': lambda x: f"{x:.6f}" if x < 1 else f"{x:.2f}"
                }))
                
                # Interpretation
                st.markdown("""
                **Interpretation:**
                - **MSE (Mean Squared Error):** Lower values indicate better fit (0 is perfect).
                - **MAE (Mean Absolute Error):** Average absolute error in the same units as the target.
                - **R² Score:** 1 is perfect prediction, 0 is no better than a horizontal line, negative if worse.
                """)
                
                # Additional insights
                st.markdown("""
                **Model Insights:**
                - The model has been trained on {} days of historical data.
                - The test set contains {} days of data.
                - The forecast extends {} trading days into the future.
                """.format(len(X_train), len(X_test), forecast_days))
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
else:
    # Display instructions when the app first loads
    st.info("""
    ### How to use this app:
    1. Select a stock ticker from the dropdown in the sidebar
    2. Choose the number of years of historical data to use
    3. Select how many days into the future you want to forecast
    4. Adjust the model parameters if needed (optional)
    5. Click the 'Run Analysis' button to start the prediction
    
    The app will fetch the historical stock data, preprocess it, train the model, 
    and display the prediction results.
    """)
    st.markdown("### Available Tickers")
    st.write("Here are some popular stocks you can analyze:")
    
    # Create a grid of tickers with their names
    ticker_info = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc. (Google)',
        'AMZN': 'Amazon.com Inc.',
        'TSLA': 'Tesla Inc.',
        'META': 'Meta Platforms Inc. (Facebook)',
        'NVDA': 'NVIDIA Corporation',
        'JPM': 'JPMorgan Chase & Co.',
        'JNJ': 'Johnson & Johnson',
        'WMT': 'Walmart Inc.'
    }
    
    # Display tickers in a nice grid
    cols = st.columns(2)  # 2 columns
    for i, (ticker, name) in enumerate(ticker_info.items()):
        with cols[i % 2]:
            st.markdown(f"**{ticker}** - {name}")
    
    # Add a note about data source
    st.markdown("""
    ---
    *Note: Stock data is provided by Yahoo Finance. The predictions are for educational purposes only and should not be considered as financial advice.*
    """)
