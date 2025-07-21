import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import os
import time
from data_utils import load_or_fetch_data, preprocess_data, train_test_split, DEFAULT_TICKERS
from model_utils import build_lstm_model, train_model, predict_future, load_or_train_model, cross_validate

# Set page config
st.set_page_config(
    page_title="StockSense AI - Advanced Stock Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1d391kg:focus:not(:focus-visible) {
        background-color: #0A0D14;
        border-right: 1px solid #2D3748;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Cards */
    .metric-card {
        background: #1E293B;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #1E293B;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        margin: 0 2px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: #4CAF50;
        color: white !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0E1117;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4CAF50;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.icons8.com/color/96/000000/stock-share.png", width=80)
with col2:
    st.title("StockSense AI")
    st.caption("Advanced AI-powered stock price prediction and analysis")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## 🔍 Stock Selection")
    
    # Ticker selection with search
    ticker = st.selectbox(
        "Select a stock ticker:",
        DEFAULT_TICKERS,
        index=0,
        key="ticker_select"
    )
    
    # Quick select buttons for popular stocks
    st.markdown("<div style='margin: 10px 0; font-size: 14px; color: #94A3B8;'>Popular stocks:</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    popular_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META"]
    for i, t in enumerate(popular_tickers):
        with cols[i % 3]:
            if st.button(t):
                ticker = t
    
    # Data range selection
    st.markdown("---")
    st.markdown("## 📅 Data Range")
    
    years = st.slider(
        "Years of historical data:",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="More years provide more training data but may include outdated patterns"
    )
    
    forecast_days = st.slider(
        "Days to forecast:",
        min_value=5,
        max_value=90,
        value=30,
        step=5,
        help="Number of trading days to predict into the future"
    )
    
    # Model parameters
    st.markdown("---")
    st.markdown("## ⚙️ Model Settings")
    
    epochs = st.slider(
        "Training epochs:",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Number of training iterations (more can lead to overfitting)"
    )
    
    batch_size = st.selectbox(
        "Batch size:",
        [16, 32, 64, 128],
        index=1,
        help="Number of samples processed before the model is updated"
    )
    
    # Add some info
    st.markdown("---")
    st.markdown("""
    <div style='font-size: 12px; color: #94A3B8;'>
    <p>💡 <strong>Tip:</strong> Start with default settings and adjust based on your needs.</p>
    <p>📊 Data provided by Yahoo Finance</p>
    </div>
    """, unsafe_allow_html=True)

# Main content area
if st.sidebar.button("🚀 Run Analysis", use_container_width=True):
    with st.spinner('Fetching data and training model...'):
        try:
            # Load or fetch data
            df = load_or_fetch_data(ticker, years)
            
            if df is None or df.empty:
                st.error("Failed to fetch data. Please try again later.")
            else:
                # Preprocess data
                scaled_data, scaler, df = preprocess_data(df)
                
                # Split data
                X_train, X_test, y_train, y_test, X, y = train_test_split(scaled_data)
                
                # Train or load model
                model, history = load_or_train_model(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Inverse transform predictions and actual values
                y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                
                # Calculate metrics
                mse = np.mean((y_pred_actual - y_test_actual) ** 2)
                mae = np.mean(np.abs(y_pred_actual - y_test_actual))
                r2 = 1 - (np.sum((y_test_actual - y_pred_actual) ** 2) / 
                         np.sum((y_test_actual - np.mean(y_test_actual)) ** 2))
                
                # Predict future prices
                last_sequence = X[-1].reshape(1, X.shape[1], 1)
                future_predictions = predict_future(model, last_sequence, forecast_days, scaler)
                future_dates = pd.date_range(start=df.index[-1], periods=forecast_days+1, freq='B')[1:]
                
                # Display results
                st.balloons()
                st.success("✅ Analysis completed successfully!")
                
                # Display metrics in cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <div style="color: #94A3B8; font-size: 14px;">Current Price</div>
                        <div style="font-size: 24px; font-weight: bold; margin: 8px 0;">${:,.2f}</div>
                    </div>
                    """.format(y_test_actual[-1]), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="metric-card">
                        <div style="color: #94A3B8; font-size: 14px;">Predicted Price</div>
                        <div style="font-size: 24px; font-weight: bold; margin: 8px 0;">${:,.2f}</div>
                    </div>
                    """.format(future_predictions[0]), unsafe_allow_html=True)
                
                with col3:
                    price_change = ((future_predictions[0] - y_test_actual[-1]) / y_test_actual[-1]) * 100
                    change_color = "#10B981" if price_change >= 0 else "#EF4444"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: #94A3B8; font-size: 14px;">Forecast Change</div>
                        <div style="font-size: 24px; font-weight: bold; margin: 8px 0; color: {color};">
                            {change:+.2f}%
                        </div>
                    </div>
                    """.format(color=change_color, change=price_change), unsafe_allow_html=True)
                
                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["📈 Price Forecast", "📊 Performance Metrics", "📉 Historical Data"])
                
                with tab1:
                    # Create interactive price chart
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=df.index[-len(y_test):],
                        y=y_test_actual,
                        mode='lines',
                        name='Historical Prices',
                        line=dict(color='#4CAF50', width=2),
                        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                    ))
                    
                    # Add predicted values
                    fig.add_trace(go.Scatter(
                        x=df.index[-len(y_test):],
                        y=y_pred_actual,
                        mode='lines',
                        name='Model Predictions',
                        line=dict(color='#EF4444', width=2, dash='dash'),
                        hovertemplate='Date: %{x}<br>Predicted: $%{y:.2f}<extra></extra>'
                    ))
                    
                    # Add future predictions
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_predictions,
                        mode='lines+markers',
                        name='Future Forecast',
                        line=dict(color='#3B82F6', width=3),
                        marker=dict(size=8),
                        hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f'{ticker} Stock Price Forecast',
                        xaxis_title='Date',
                        yaxis_title='Price ($)',
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1.02,
                            xanchor='right',
                            x=1
                        ),
                        hovermode='x unified',
                        template='plotly_dark',
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=600,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#FAFAFA')
                    )
                    
                    # Add range slider and buttons
                    fig.update_xaxes(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider_visible=True,
                        type="date"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Display metrics in a nice layout
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
                    with col2:
                        st.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
                    with col3:
                        st.metric("R² Score", f"{r2:.4f}")
                    
                    # Add training history plot
                    if history:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=history.history['loss'],
                            name='Training Loss',
                            mode='lines',
                            line=dict(color='#4CAF50')
                        ))
                        if 'val_loss' in history.history:
                            fig.add_trace(go.Scatter(
                                y=history.history['val_loss'],
                                name='Validation Loss',
                                mode='lines',
                                line=dict(color='#EF4444')
                            ))
                        
                        fig.update_layout(
                            title='Model Training History',
                            xaxis_title='Epoch',
                            yaxis_title='Loss (MSE)',
                            template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#FAFAFA')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    # Display historical data in a table
                    st.dataframe(df.tail(10), use_container_width=True)
                    
                    # Add download button for the data
                    csv = df.to_csv().encode('utf-8')
                    st.download_button(
                        label="Download Full Data (CSV)",
                        data=csv,
                        file_name=f"{ticker}_historical_data.csv",
                        mime='text/csv'
                    )
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    # Show welcome message
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <h2>Welcome to StockSense AI</h2>
        <p style="color: #94A3B8; max-width: 600px; margin: 0 auto 2rem auto;">
            Get started by selecting a stock ticker and clicking the "Run Analysis" button in the sidebar.
            Our AI model will analyze historical price data and generate predictions for the selected stock.
        </p>
        <div style="margin: 2rem 0;">
            <h3>How it works:</h3>
            <ol style="text-align: left; max-width: 500px; margin: 1rem auto;">
                <li>Select a stock ticker from the sidebar</li>
                <li>Choose the historical data range and forecast period</li>
                <li>Adjust model parameters (optional)</li>
                <li>Click "Run Analysis" to train the model and generate predictions</li>
            </ol>
        </div>
        <div style="margin-top: 3rem; color: #64748B; font-size: 0.9rem;">
            <p>💡 <strong>Tip:</strong> Start with popular stocks like AAPL, MSFT, or GOOGL</p>
            <p>📊 Data provided by Yahoo Finance</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.9rem; margin-top: 2rem;">
    <p>StockSense AI - Advanced Stock Price Prediction Tool</p>
    <p>This tool is for educational purposes only and should not be considered financial advice.</p>
</div>
""", unsafe_allow_html=True)
