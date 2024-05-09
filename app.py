import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import webbrowser

# Hide the Streamlit menu and the "Deploy" button using custom CSS
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
         .stDeployButton {
    display: none;
  }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Add your logo text here
st.sidebar.markdown("<div class='stMarkdown' style='width: 288px; position: fixed; top: -25px; left: 0; padding: 20px;'><div><h1 style='text-align: left;'>PrediStock</h1></div></div>", unsafe_allow_html=True)

st.title('Stock Forecast System')

# Sidebar with the slider for years of prediction
sidebar_html = """
    <style>
        .sidebar .sidebar-content {
            width: 350px;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: auto;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 2rem;
            height: 100vh;
        }
    </style>
"""
st.sidebar.markdown(sidebar_html, unsafe_allow_html=True)

st.sidebar.title('Settings')
n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365

custom_stock = st.text_input('Enter stock symbol (e.g., TSLA)')
stock_symbol = custom_stock.upper() if custom_stock else None

start_date = st.sidebar.date_input('Start date', date(2015, 1, 1))
end_date = st.sidebar.date_input('End date', date.today())

# Home button to return to main page
st.sidebar.title('Settings')
if st.sidebar.button('Home'):
    webbrowser.open('file:///C:/sem%206%20project%20ganesh/stockprediction/Predistock/index.html')

if stock_symbol:
    try:
        stock_info = yf.Ticker(stock_symbol).info

        st.subheader(f"{stock_info['shortName']} ({stock_symbol})")
        st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")

        # Fetch current price
        current_price_data = yf.Ticker(stock_symbol).history(period='1d')
        if not current_price_data.empty:
            current_price = current_price_data.iloc[-1]['Close']
            st.write(f"**Current Price:** {stock_info.get('currency', '')} {current_price}")
        else:
            st.warning("Current price information is not available for this stock.")

        st.write(f"**Market Cap:** {stock_info.get('currency', '')} {stock_info.get('marketCap', 'N/A')}")
        st.write(f"**Description:** {stock_info.get('longBusinessSummary', 'N/A')[:400]}...")  # Limiting to 400 characters

        @st.cache_data
        def load_data(ticker, start, end):
            data = yf.download(ticker, start, end)
            data.reset_index(inplace=True)
            return data

        data_load_state = st.text('Loading data...')
        data = load_data(stock_symbol, start_date, end_date)
        data_load_state.text('Loading data... done!')

        st.subheader('Raw data')
        st.write(data.tail())

        # Plot raw data
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        plot_raw_data()

        # Predict forecast with Prophet.
        df_train = data[['Date','Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Show and plot forecast
        st.subheader('Forecast data')
        st.write(forecast.tail())

        st.write(f'Forecast plot for {n_years} years')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.write("Forecast components")
        fig2 = m.plot_components(forecast)
        st.write(fig2)

        # Calculate profit/loss and recommendation
        current_price = data.iloc[-1]['Close']
        initial_price = data.iloc[0]['Close']
        price_difference = current_price - initial_price

        if price_difference > 0:
            st.success(f"The stock has shown a profit of {price_difference:.2f} . You may consider buying.")
        elif price_difference < 0:
            st.error(f"The stock has shown a loss of {abs(price_difference):.2f} . You may consider not buying.")
        else:
            st.warning("The stock price remains unchanged. Consider evaluating other factors before making a decision.")
    except Exception as e:
        st.error(f"An error occurred while fetching stock information: {str(e)}")
else:
    st.warning("Please enter a valid stock symbol.")
