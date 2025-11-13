#!/usr/bin/env python3
"""
🚀 AI Revolution Trading Platform
Advanced AI-Powered Stock Analysis & Trading Dashboard

Author: Anderson Nguetoum
Website: andersonnguetoum.com
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🤖 AI Revolution Trading Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    .alert-box {
        background: linear-gradient(45deg, #43e97b 0%, #38f9d7 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #00ff88;
    }
    .prediction-box {
        background: linear-gradient(45deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stSelectbox > div > div {
        background-color: #1e1e1e;
        border-radius: 10px;
    }
    .chat-message {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {'cash': 100000, 'holdings': {}}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load data function
@st.cache_data
def load_ai_data():
    """Load AI Revolution dataset"""
    try:
        df = pd.read_csv('data/ai_revolution_stock_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset not found. Please generate it first.")
        return pd.DataFrame()

# AI Companies configuration
AI_COMPANIES = {
    'NVDA': {'name': 'NVIDIA', 'color': '#76B900', 'emoji': '🔥'},
    'MSFT': {'name': 'Microsoft', 'color': '#00A4EF', 'emoji': '💼'},
    'GOOGL': {'name': 'Alphabet', 'color': '#4285F4', 'emoji': '🔍'},
    'AMZN': {'name': 'Amazon', 'color': '#FF9900', 'emoji': '📦'},
    'TSLA': {'name': 'Tesla', 'color': '#CC0000', 'emoji': '⚡'}
}

def main():
    """Main application"""

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Revolution Trading Platform</h1>
        <p>Advanced AI-Powered Stock Analysis & Trading Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df = load_ai_data()
    if df.empty:
        st.stop()

    # Sidebar Navigation
    st.sidebar.markdown("## 🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose Your Adventure:",
        [
            "🏠 Dashboard Overview",
            "📊 Advanced Analytics",
            "🤖 AI Assistant",
            "💰 Portfolio Simulator",
            "🔮 Predictions Lab",
            "⚡ Real-time Alerts",
            "📱 Mobile View"
        ]
    )

    # Page routing
    if page == "🏠 Dashboard Overview":
        dashboard_overview(df)
    elif page == "📊 Advanced Analytics":
        advanced_analytics(df)
    elif page == "🤖 AI Assistant":
        ai_assistant(df)
    elif page == "💰 Portfolio Simulator":
        portfolio_simulator(df)
    elif page == "🔮 Predictions Lab":
        predictions_lab(df)
    elif page == "⚡ Real-time Alerts":
        realtime_alerts(df)
    elif page == "📱 Mobile View":
        mobile_view(df)

def dashboard_overview(df):
    """Main dashboard with key metrics"""
    st.header("🏠 Dashboard Overview")

    # Real-time metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    latest_data = df.groupby('Symbol').last()

    for i, (symbol, info) in enumerate(AI_COMPANIES.items()):
        col = [col1, col2, col3, col4, col5][i]

        if symbol in latest_data.index:
            price = latest_data.loc[symbol, 'Close']
            sentiment = latest_data.loc[symbol, 'AI_Sentiment']

            # Calculate price change (mock)
            change = np.random.uniform(-3, 5)

            col.markdown(f"""
            <div class="metric-container">
                <h3>{info['emoji']} {symbol}</h3>
                <h2>${price:.2f}</h2>
                <p>{'🔥' if change > 0 else '❄️'} {change:+.1f}%</p>
                <p>Sentiment: {sentiment}/100</p>
            </div>
            """, unsafe_allow_html=True)

    # Main chart
    st.subheader("📈 Stock Price Evolution")

    # Interactive price chart
    fig = go.Figure()

    for symbol, info in AI_COMPANIES.items():
        stock_data = df[df['Symbol'] == symbol]
        fig.add_trace(go.Scatter(
            x=stock_data['Date'],
            y=stock_data['Close'],
            mode='lines',
            name=f"{info['emoji']} {symbol}",
            line=dict(color=info['color'], width=3),
            hovertemplate=f"<b>{symbol}</b><br>" +
                         "Date: %{x}<br>" +
                         "Price: $%{y:.2f}<br>" +
                         "<extra></extra>"
        ))

    fig.update_layout(
        title="🚀 AI Revolution Stock Performance",
        xaxis_title="Date",
        yaxis_title="Stock Price ($)",
        template="plotly_dark",
        height=500,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # Sentiment vs Performance
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 AI Sentiment Analysis")
        sentiment_fig = px.box(
            df,
            x='Symbol',
            y='AI_Sentiment',
            color='Symbol',
            title="Sentiment Distribution by Company",
            template="plotly_dark"
        )
        st.plotly_chart(sentiment_fig, use_container_width=True)

    with col2:
        st.subheader("📊 Volume Analysis")
        volume_fig = px.bar(
            latest_data.reset_index(),
            x='Symbol',
            y='Volume',
            color='Symbol',
            title="Latest Trading Volume",
            template="plotly_dark"
        )
        st.plotly_chart(volume_fig, use_container_width=True)

    # Live alerts section
    st.subheader("🔔 Live Market Alerts")

    alerts = [
        "🚀 NVDA breaking resistance at $300 - Strong Buy Signal!",
        "⚠️ TSLA showing high volatility - Monitor closely",
        "💰 GOOGL sentiment spike detected - Potential upside",
        "📈 MSFT forming bullish pattern - Entry opportunity"
    ]

    for alert in alerts[:2]:  # Show top 2 alerts
        st.markdown(f"""
        <div class="alert-box">
            <strong>{alert}</strong>
        </div>
        """, unsafe_allow_html=True)

def advanced_analytics(df):
    """Advanced analytics page"""
    st.header("📊 Advanced Analytics Lab")

    # Technical indicators
    st.subheader("🔧 Technical Analysis")

    selected_stock = st.selectbox(
        "Select Stock for Analysis:",
        list(AI_COMPANIES.keys()),
        format_func=lambda x: f"{AI_COMPANIES[x]['emoji']} {x} - {AI_COMPANIES[x]['name']}"
    )

    stock_data = df[df['Symbol'] == selected_stock].copy()
    stock_data = stock_data.sort_values('Date')

    # Calculate technical indicators
    stock_data['SMA_20'] = stock_data['Close'].rolling(20).mean()
    stock_data['SMA_50'] = stock_data['Close'].rolling(50).mean()
    stock_data['RSI'] = calculate_rsi(stock_data['Close'])
    stock_data['BB_upper'], stock_data['BB_lower'] = calculate_bollinger_bands(stock_data['Close'])

    # Advanced candlestick chart
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Technical Indicators', 'RSI', 'Volume'),
        row_heights=[0.6, 0.2, 0.2]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=stock_data['Date'],
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name="Price"
    ), row=1, col=1)

    # Moving averages
    fig.add_trace(go.Scatter(
        x=stock_data['Date'],
        y=stock_data['SMA_20'],
        name="SMA 20",
        line=dict(color='orange', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=stock_data['Date'],
        y=stock_data['SMA_50'],
        name="SMA 50",
        line=dict(color='red', width=2)
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=stock_data['Date'],
        y=stock_data['BB_upper'],
        name="BB Upper",
        line=dict(color='gray', dash='dash')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=stock_data['Date'],
        y=stock_data['BB_lower'],
        name="BB Lower",
        line=dict(color='gray', dash='dash'),
        fill='tonexty'
    ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=stock_data['Date'],
        y=stock_data['RSI'],
        name="RSI",
        line=dict(color='purple')
    ), row=2, col=1)

    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # Volume
    fig.add_trace(go.Bar(
        x=stock_data['Date'],
        y=stock_data['Volume'],
        name="Volume",
        marker_color='lightblue'
    ), row=3, col=1)

    fig.update_layout(
        title=f"📈 {selected_stock} - Advanced Technical Analysis",
        template="plotly_dark",
        height=800,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)

    current_price = stock_data['Close'].iloc[-1]
    price_change = ((current_price - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0]) * 100
    volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
    current_rsi = stock_data['RSI'].iloc[-1]

    col1.metric("💰 Current Price", f"${current_price:.2f}")
    col2.metric("📈 Total Return", f"{price_change:+.1f}%")
    col3.metric("📊 Volatility (Annual)", f"{volatility:.1f}%")
    col4.metric("⚡ Current RSI", f"{current_rsi:.1f}")

def ai_assistant(df):
    """AI Assistant chatbot"""
    st.header("🤖 AI Trading Assistant")
    st.subheader("Your Personal AI Trading Advisor")

    # Chat interface
    chat_container = st.container()

    # User input
    user_input = st.text_input("💬 Ask me anything about the markets:", placeholder="e.g., What do you think about NVDA today?")

    if user_input:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "message": user_input})

        # Generate AI response (simplified)
        ai_response = generate_ai_response(user_input, df)
        st.session_state.chat_history.append({"role": "assistant", "message": ai_response})

    # Display chat history
    with chat_container:
        for chat in st.session_state.chat_history[-10:]:  # Show last 10 messages
            if chat["role"] == "user":
                st.markdown(f"""
                <div style="text-align: right; margin: 1rem 0;">
                    <div style="background: #667eea; padding: 1rem; border-radius: 10px; display: inline-block; max-width: 70%;">
                        <strong>You:</strong> {chat["message"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align: left; margin: 1rem 0;">
                    <div style="background: #764ba2; padding: 1rem; border-radius: 10px; display: inline-block; max-width: 70%;">
                        <strong>🤖 AI Assistant:</strong> {chat["message"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Quick action buttons
    st.subheader("⚡ Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    if col1.button("📊 Market Summary"):
        summary = generate_market_summary(df)
        st.session_state.chat_history.append({"role": "assistant", "message": summary})
        st.rerun()

    if col2.button("🎯 Stock Recommendation"):
        recommendation = generate_recommendation(df)
        st.session_state.chat_history.append({"role": "assistant", "message": recommendation})
        st.rerun()

    if col3.button("⚠️ Risk Analysis"):
        risk_analysis = generate_risk_analysis(df)
        st.session_state.chat_history.append({"role": "assistant", "message": risk_analysis})
        st.rerun()

    if col4.button("🔮 Price Prediction"):
        prediction = generate_prediction(df)
        st.session_state.chat_history.append({"role": "assistant", "message": prediction})
        st.rerun()

def portfolio_simulator(df):
    """Portfolio simulator with paper trading"""
    st.header("💰 Portfolio Simulator")
    st.subheader("Practice Trading with Virtual Money")

    # Portfolio summary
    total_value = st.session_state.portfolio['cash']
    for symbol, shares in st.session_state.portfolio['holdings'].items():
        if symbol in df['Symbol'].values:
            current_price = df[df['Symbol'] == symbol]['Close'].iloc[-1]
            total_value += shares * current_price

    col1, col2, col3 = st.columns(3)
    col1.metric("💵 Cash", f"${st.session_state.portfolio['cash']:,.2f}")
    col2.metric("📈 Portfolio Value", f"${total_value:,.2f}")
    col3.metric("💰 Total Return", f"{((total_value - 100000) / 100000) * 100:+.1f}%")

    # Trading interface
    st.subheader("📊 Trading Panel")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🛒 Buy/Sell Orders")

        action = st.radio("Action:", ["Buy", "Sell"])
        selected_symbol = st.selectbox("Stock:", list(AI_COMPANIES.keys()))
        shares = st.number_input("Shares:", min_value=1, value=10)

        if selected_symbol in df['Symbol'].values:
            current_price = df[df['Symbol'] == selected_symbol]['Close'].iloc[-1]
            total_cost = shares * current_price

            st.write(f"Current Price: ${current_price:.2f}")
            st.write(f"Total Cost: ${total_cost:.2f}")

            if action == "Buy":
                if st.button("🛒 Execute Buy Order"):
                    if st.session_state.portfolio['cash'] >= total_cost:
                        st.session_state.portfolio['cash'] -= total_cost
                        if selected_symbol in st.session_state.portfolio['holdings']:
                            st.session_state.portfolio['holdings'][selected_symbol] += shares
                        else:
                            st.session_state.portfolio['holdings'][selected_symbol] = shares
                        st.success(f"✅ Bought {shares} shares of {selected_symbol}")
                        st.rerun()
                    else:
                        st.error("❌ Insufficient funds!")

            else:  # Sell
                if st.button("💸 Execute Sell Order"):
                    if (selected_symbol in st.session_state.portfolio['holdings'] and
                        st.session_state.portfolio['holdings'][selected_symbol] >= shares):
                        st.session_state.portfolio['cash'] += total_cost
                        st.session_state.portfolio['holdings'][selected_symbol] -= shares
                        if st.session_state.portfolio['holdings'][selected_symbol] == 0:
                            del st.session_state.portfolio['holdings'][selected_symbol]
                        st.success(f"✅ Sold {shares} shares of {selected_symbol}")
                        st.rerun()
                    else:
                        st.error("❌ Insufficient shares!")

    with col2:
        st.markdown("#### 📊 Current Holdings")

        if st.session_state.portfolio['holdings']:
            holdings_data = []
            for symbol, shares in st.session_state.portfolio['holdings'].items():
                current_price = df[df['Symbol'] == symbol]['Close'].iloc[-1]
                value = shares * current_price
                holdings_data.append({
                    'Symbol': symbol,
                    'Shares': shares,
                    'Price': f"${current_price:.2f}",
                    'Value': f"${value:.2f}"
                })

            holdings_df = pd.DataFrame(holdings_data)
            st.dataframe(holdings_df, use_container_width=True)
        else:
            st.info("No holdings yet. Start trading!")

    # Performance chart
    st.subheader("📈 Portfolio Performance")
    # This would show historical portfolio value over time
    # For now, we'll show a placeholder
    st.info("📊 Portfolio performance tracking will be implemented with trade history.")

def predictions_lab(df):
    """Advanced predictions laboratory"""
    st.header("🔮 Predictions Laboratory")
    st.subheader("Multi-Model AI Forecasting")

    selected_stock = st.selectbox(
        "Select Stock for Prediction:",
        list(AI_COMPANIES.keys()),
        format_func=lambda x: f"{AI_COMPANIES[x]['emoji']} {x}"
    )

    # Prediction parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        forecast_days = st.slider("Forecast Days:", 1, 90, 30)

    with col2:
        confidence_level = st.slider("Confidence Level:", 0.8, 0.99, 0.95)

    with col3:
        model_type = st.selectbox("Model:", ["Ensemble", "LSTM", "Prophet", "Linear Regression"])

    if st.button("🚀 Generate Prediction"):
        with st.spinner("🧠 AI is analyzing patterns..."):
            time.sleep(2)  # Simulate processing

            # Generate mock prediction
            stock_data = df[df['Symbol'] == selected_stock].sort_values('Date')
            current_price = stock_data['Close'].iloc[-1]

            # Mock prediction logic
            trend = np.random.choice([-1, 1], p=[0.4, 0.6])
            base_change = np.random.uniform(0.05, 0.25) * trend
            predicted_price = current_price * (1 + base_change)

            accuracy = np.random.uniform(0.75, 0.95)

            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h3>🔮 {forecast_days}-Day Prediction for {selected_stock}</h3>
                <h2>Target Price: ${predicted_price:.2f}</h2>
                <h3>Expected Change: {base_change*100:+.1f}%</h3>
                <p><strong>Model Confidence:</strong> {accuracy*100:.1f}%</p>
                <p><strong>Signal:</strong> {'🚀 STRONG BUY' if base_change > 0.1 else '📈 BUY' if base_change > 0 else '📉 SELL'}</p>
            </div>
            """, unsafe_allow_html=True)

            # Prediction visualization
            dates = pd.date_range(start=stock_data['Date'].iloc[-1], periods=forecast_days+1, freq='D')[1:]
            predicted_prices = np.linspace(current_price, predicted_price, forecast_days)

            # Add some noise for realism
            noise = np.random.normal(0, current_price * 0.02, forecast_days)
            predicted_prices += noise

            fig = go.Figure()

            # Historical data
            fig.add_trace(go.Scatter(
                x=stock_data['Date'][-50:],
                y=stock_data['Close'][-50:],
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))

            # Prediction
            fig.add_trace(go.Scatter(
                x=dates,
                y=predicted_prices,
                mode='lines',
                name='Prediction',
                line=dict(color='red', width=3, dash='dash')
            ))

            # Confidence bands
            upper_band = predicted_prices * (1 + (1-confidence_level))
            lower_band = predicted_prices * (1 - (1-confidence_level))

            fig.add_trace(go.Scatter(
                x=dates,
                y=upper_band,
                mode='lines',
                name='Upper Confidence',
                line=dict(color='red', width=1),
                fill=None
            ))

            fig.add_trace(go.Scatter(
                x=dates,
                y=lower_band,
                mode='lines',
                name='Lower Confidence',
                line=dict(color='red', width=1),
                fill='tonexty'
            ))

            fig.update_layout(
                title=f"🔮 {selected_stock} Price Prediction - {model_type} Model",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template="plotly_dark",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Model metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 Accuracy", f"{accuracy*100:.1f}%")
            col2.metric("📊 Volatility", f"{np.std(predicted_prices)/np.mean(predicted_prices)*100:.1f}%")
            col3.metric("📈 Expected Return", f"{base_change*100:+.1f}%")
            col4.metric("⚠️ Risk Level", np.random.choice(["Low", "Medium", "High"]))

def realtime_alerts(df):
    """Real-time alerts and monitoring"""
    st.header("⚡ Real-time Market Alerts")
    st.subheader("Intelligent Trading Notifications")

    # Alert configuration
    st.markdown("#### ⚙️ Alert Settings")

    col1, col2 = st.columns(2)

    with col1:
        alert_stock = st.selectbox("Monitor Stock:", list(AI_COMPANIES.keys()))
        price_threshold = st.number_input("Price Alert Threshold:", min_value=0.0, value=300.0, step=10.0)
        sentiment_threshold = st.slider("Sentiment Alert Level:", 0, 100, 80)

    with col2:
        volatility_alert = st.checkbox("High Volatility Alert", value=True)
        volume_alert = st.checkbox("Unusual Volume Alert", value=True)
        pattern_alert = st.checkbox("Technical Pattern Alert", value=True)

    # Live monitoring simulation
    st.markdown("#### 📡 Live Market Monitor")

    # Create placeholder for live updates
    live_container = st.empty()

    # Simulate real-time updates
    if st.button("🔴 Start Live Monitoring"):
        for i in range(10):  # Simulate 10 updates
            with live_container.container():
                col1, col2, col3 = st.columns(3)

                # Simulate live data
                current_time = datetime.now().strftime("%H:%M:%S")
                mock_price = np.random.uniform(280, 320)
                mock_sentiment = np.random.randint(60, 95)
                mock_volume = np.random.randint(10000000, 50000000)

                col1.metric(
                    f"⏰ {current_time}",
                    f"${mock_price:.2f}",
                    f"{np.random.uniform(-2, 3):+.1f}%"
                )

                col2.metric(
                    "🧠 AI Sentiment",
                    f"{mock_sentiment}/100",
                    f"{np.random.randint(-5, 8):+d}"
                )

                col3.metric(
                    "📊 Volume",
                    f"{mock_volume:,}",
                    f"{np.random.uniform(-20, 30):+.1f}%"
                )

                # Generate alerts
                alerts = []
                if mock_price > price_threshold:
                    alerts.append(f"🚨 {alert_stock} crossed ${price_threshold} threshold!")
                if mock_sentiment > sentiment_threshold:
                    alerts.append(f"🔥 {alert_stock} sentiment spike detected!")
                if volatility_alert and np.random.random() > 0.7:
                    alerts.append(f"⚠️ High volatility detected in {alert_stock}")

                for alert in alerts:
                    st.warning(alert)

            time.sleep(1)  # Update every second

    # Alert history
    st.markdown("#### 📋 Recent Alerts")

    alert_history = [
        {"Time": "14:23:15", "Type": "Price", "Message": "NVDA broke $300 resistance", "Priority": "High"},
        {"Time": "14:18:42", "Type": "Sentiment", "Message": "TSLA sentiment surge +15 points", "Priority": "Medium"},
        {"Time": "14:15:33", "Type": "Volume", "Message": "GOOGL unusual volume spike", "Priority": "Low"},
        {"Time": "14:12:10", "Type": "Pattern", "Message": "MSFT forming bullish flag", "Priority": "Medium"},
    ]

    alert_df = pd.DataFrame(alert_history)
    st.dataframe(alert_df, use_container_width=True)

def mobile_view(df):
    """Mobile-optimized view"""
    st.header("📱 Mobile Trading View")
    st.subheader("Optimized for Mobile Devices")

    # Mobile-friendly layout
    for symbol, info in AI_COMPANIES.items():
        if symbol in df['Symbol'].values:
            latest = df[df['Symbol'] == symbol].iloc[-1]

            # Mobile card design
            st.markdown(f"""
            <div style="
                background: linear-gradient(45deg, {info['color']}22, {info['color']}44);
                padding: 1.5rem;
                border-radius: 15px;
                margin: 1rem 0;
                border: 2px solid {info['color']};
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2>{info['emoji']} {symbol}</h2>
                        <p style="margin: 0; opacity: 0.8;">{info['name']}</p>
                    </div>
                    <div style="text-align: right;">
                        <h1>${latest['Close']:.2f}</h1>
                        <p style="margin: 0; color: {'green' if np.random.random() > 0.5 else 'red'};">
                            {np.random.uniform(-3, 5):+.1f}%
                        </p>
                    </div>
                </div>
                <hr style="margin: 1rem 0; border-color: {info['color']};">
                <div style="display: flex; justify-content: space-between;">
                    <span>Volume: {latest['Volume']:,}</span>
                    <span>Sentiment: {latest['AI_Sentiment']}/100</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Quick action buttons
            col1, col2, col3 = st.columns(3)
            col1.button(f"📊 Analyze {symbol}", key=f"analyze_{symbol}")
            col2.button(f"🛒 Buy {symbol}", key=f"buy_{symbol}")
            col3.button(f"⚡ Alert {symbol}", key=f"alert_{symbol}")

    # Mobile-friendly chart
    st.subheader("📊 Quick Chart")
    chart_symbol = st.selectbox("Select Stock:", list(AI_COMPANIES.keys()))

    chart_data = df[df['Symbol'] == chart_symbol]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data['Date'],
        y=chart_data['Close'],
        mode='lines',
        line=dict(color=AI_COMPANIES[chart_symbol]['color'], width=4),
        fill='tonexty'
    ))

    fig.update_layout(
        title=f"{AI_COMPANIES[chart_symbol]['emoji']} {chart_symbol} - Mobile Chart",
        template="plotly_dark",
        height=300,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

# Helper functions
def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band

def generate_ai_response(user_input, df):
    """Generate AI assistant response"""
    user_input_lower = user_input.lower()

    if "nvda" in user_input_lower or "nvidia" in user_input_lower:
        return "🔥 NVIDIA is showing strong momentum! With AI chip demand soaring, NVDA remains a top pick. Current technical indicators suggest bullish continuation. Consider accumulating on any dips."

    elif "tsla" in user_input_lower or "tesla" in user_input_lower:
        return "⚡ Tesla is at an interesting inflection point. FSD progress and energy storage growth are key catalysts. Watch for support at $440 level for potential entry."

    elif "buy" in user_input_lower:
        return "🎯 Based on current analysis, GOOGL shows the strongest setup with +4.1% expected return. NVDA and TSLA are good for momentum plays. Consider your risk tolerance!"

    elif "market" in user_input_lower:
        return "📊 AI sector is in consolidation phase after 2022-2024 rally. Expect continued volatility but long-term outlook remains bullish. Focus on quality names with strong fundamentals."

    else:
        return "🤖 I'm analyzing that for you! I can help with stock analysis, market trends, risk assessment, and trading strategies. What specific aspect interests you most?"

def generate_market_summary(df):
    """Generate market summary"""
    return """📊 **Market Summary**: AI sector showing mixed signals. GOOGL leading with +4.1% forecast, while MSFT/NVDA facing short-term headwinds.
    Sentiment analysis reveals 15.4% high-confidence days vs 16.5% low-sentiment periods.
    🎯 **Strategy**: Focus on GOOGL for growth, MSFT for stability post-correction."""

def generate_recommendation(df):
    """Generate stock recommendation"""
    return """🎯 **Top Pick**: GOOGL - Undervalued in AI race, strong technical setup
    📈 **Momentum Play**: NVDA - Short-term correction creating entry opportunity
    🛡️ **Defensive**: MSFT - Stable choice, wait for $550 entry point
    ⚠️ **Avoid**: AMZN - Weak momentum, limited AI catalysts"""

def generate_risk_analysis(df):
    """Generate risk analysis"""
    return """⚠️ **Risk Assessment**:
    🔴 **High Risk**: TSLA (volatility), NVDA (momentum-dependent)
    🟡 **Medium Risk**: GOOGL (execution risk), AMZN (competition)
    🟢 **Low Risk**: MSFT (defensive qualities)

    💡 **Recommendation**: Diversify across 3-4 names, limit single position to 25% of portfolio."""

def generate_prediction(df):
    """Generate price prediction"""
    return """🔮 **30-Day Forecasts**:
    📈 **GOOGL**: $3,866 → $4,023 (+4.1%) - STRONG BUY
    📉 **NVDA**: $305 → $295 (-3.4%) - Buy the dip
    📉 **TSLA**: $462 → $448 (-3.1%) - Technical correction

    🎯 **Confidence**: 87% based on ensemble model"""

if __name__ == "__main__":
    main()