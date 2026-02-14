import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import streamlit.components.v1 as components
import webbrowser
import pandas as pd

# Hide Streamlit menu & deploy button
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('Stock Forecast System')

# ---------------- SIDEBAR ----------------

st.sidebar.markdown("""
<h1 style='margin-bottom:5px;'>PrediStock</h1>
""", unsafe_allow_html=True)

st.sidebar.title('Settings')

n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365

chart_type = st.sidebar.selectbox(
    "Select Chart Type",
    ("Line Chart", "Candlestick")
)

# Default UNTICKED
show_raw = st.sidebar.checkbox("Show Raw Data", value=False)
show_confidence = st.sidebar.checkbox("Show Forecast Confidence", value=False)

if st.sidebar.button('Home'):
    webbrowser.open('https://predistocks.netlify.app/')

# ---------------- STOCK SEARCH ----------------

indian_stocks = {
    "TCS": "TCS.NS",
    "TATA CONSULTANCY": "TCS.NS",
    "RELIANCE": "RELIANCE.NS",
    "INFOSYS": "INFY.NS",
    "INFY": "INFY.NS",
    "HDFC": "HDFCBANK.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI": "ICICIBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "STATE BANK": "SBIN.NS"
}

custom_stock = st.text_input(
    'Enter Company Name or Symbol (e.g., TCS, Reliance, INFY)'
)

stock_symbol = None

if custom_stock:
    user_input = custom_stock.upper()
    stock_symbol = indian_stocks.get(user_input, user_input)

if stock_symbol:
    try:
        stock = yf.Ticker(stock_symbol)
        stock_info = stock.info

        st.subheader(f"{stock_info.get('shortName', stock_symbol)}")

        st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")

        # Current Price
        current_price_data = stock.history(period='1d')
        if not current_price_data.empty:
            current_price = current_price_data.iloc[-1]['Close']
            st.write(f"### Current Price: ₹ {current_price:,.2f}")
        else:
            st.warning("Current price not available.")

        market_cap = stock_info.get('marketCap')
        if market_cap:
            st.write(f"**Market Cap:** ₹ {market_cap:,.0f}")

        st.write(
            f"**Description:** "
            f"{stock_info.get('longBusinessSummary', 'N/A')[:400]}..."
        )

        # -------- LOAD DATA --------
        @st.cache_data
        def load_data(ticker):
            data = yf.download(ticker, period="max")
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.reset_index(inplace=True)
            data = data.dropna()
            return data

        data_load_state = st.text('Loading data...')
        data = load_data(stock_symbol)
        data_load_state.text('Loading data... done!')

        if data.empty:
            st.error("No data found.")
            st.stop()

        # Raw Data (Hidden by default)
        if show_raw:
            st.subheader('Raw Data')
            st.write(data.tail())

        # -------- CHART --------
        st.subheader('Time Series Data')

        fig = go.Figure()

        if chart_type == "Line Chart":
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Open'],
                name="Open"
            ))
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Close'],
                name="Close"
            ))
        else:
            fig.add_trace(go.Candlestick(
                x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Candlestick"
            ))

        fig.update_layout(
            title='Stock Price Chart',
            xaxis_rangeslider_visible=True
        )

        st.plotly_chart(fig)

        # -------- FORECAST --------
        df_train = data[['Date', 'Close']].copy()
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
        df_train = df_train.dropna()

        model = Prophet()
        model.fit(df_train)

        future = model.make_future_dataframe(periods=period)
        forecast = model.predict(future)

        st.subheader('Forecast Data')
        st.write(forecast.tail())

        st.write(f'Forecast plot for {n_years} years')

        if show_confidence:
            fig1 = plot_plotly(model, forecast)
        else:
            forecast_no_conf = forecast.copy()
            forecast_no_conf['yhat_upper'] = forecast_no_conf['yhat']
            forecast_no_conf['yhat_lower'] = forecast_no_conf['yhat']
            fig1 = plot_plotly(model, forecast_no_conf)

        st.plotly_chart(fig1)

        st.subheader("Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        # -------- PROFESSIONAL INVESTMENT LOGIC --------
        current_price = data.iloc[-1]['Close']
        initial_price = data.iloc[0]['Close']
        diff = current_price - initial_price
        percent_change = (diff / initial_price) * 100

        st.subheader("Investment Insight")

        if diff > 0:
            st.success(
                f"The stock has demonstrated positive growth of ₹ {diff:,.2f} "
                f"({percent_change:.2f}%) over the selected period. "
                "Based on current trend analysis, this stock shows upward momentum "
                "and may be considered for investment."
            )

            # Zerodha Redirect (FIXED)
            if st.button(f"Buy {stock_symbol}"):
                zerodha_symbol = stock_symbol.replace(".NS", "")
                zerodha_url = f"https://kite.zerodha.com/?symbol=NSE:{zerodha_symbol}"

                components.html(
                    f"""
                    <script>
                        window.open("{zerodha_url}", "_blank");
                    </script>
                    """,
                    height=0,
                )

        elif diff < 0:
            st.error(
                f"The stock has declined by ₹ {abs(diff):,.2f} "
                f"({abs(percent_change):.2f}%). "
                "Current trend indicates weakness. It is advisable "
                "to monitor the stock and wait for recovery signals "
                "before investing."
            )

        else:
            st.warning(
                "The stock price shows minimal change over the selected period. "
                "Consider observing the market for clearer signals."
            )

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.warning("Please enter a company name or stock symbol.")


