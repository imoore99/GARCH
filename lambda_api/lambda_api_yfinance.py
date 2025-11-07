

### Import libraries
import numpy as np
import pandas as pd
from datetime import timedelta
import yfinance as yf
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

### Helper functions
#function to transform date to desired format
def date_format(date_str):
    return pd.to_datetime(date_str).strftime('%Y-%m-%d')

## Function to load market data
def load_market_data(ticker, start_date, end_date):
    """
    Load market data for specified tickers within a date range.
    Returns a DataFrame with adjusted close prices.
    """
    #print(f"Loading market data from {start_date} to {end_date}") <--Not sure I need to print this
    # Load market data
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data = pd.Series(data['Close'][ticker][:], index=data.index)
    
    close = pd.DataFrame(data)
    returns = close.pct_change().dropna() * 100  # Convert to percentage
    return returns

#function to run GARCH volatility model
def run_garch_model(returns_data):
    """
    Fit GARCH model and return volatility forecast
    """
    # Fit GARCH(1,1) model
    model = arch_model(returns_data, vol='Garch', p=1, q=1, dist='t')
    fitted_model = model.fit(disp='off')
    
    # Get 1-day forecast (your existing method)
    single_day_forecast = fitted_model.forecast(horizon=1)
    base_daily_vol = np.sqrt(single_day_forecast.variance.iloc[-1, 0])
    
    return {
        'volatility_forecast': base_daily_vol,
        'annualized_volatility': base_daily_vol * np.sqrt(252),
        'date': returns_data.index[-1],
        'forecast_date': returns_data.index[-1] + pd.Timedelta(days=1)
    }

#function to compile GARCH forecast information
def compile_garch_forecasts(ticker, start_date, end_date, compiled_data, garch_data):
    #print(date_format(new_start_date), date_format(new_filter_date), date_format(new_end_date))

    # executing the function to load market data
    returns = load_market_data(ticker, start_date, end_date)

    ticker_data = {
            'Ticker': ticker,
            'Start_Date': date_format(start_date),
            'End_Date': date_format(end_date),
            'Mean_Return': returns[ticker].mean(),
            'Volatility': returns[ticker].std(),
            'Max_Return': returns[ticker].max(),
            'Min_Return': returns[ticker].min(),
            'kurtosis': returns[ticker].kurtosis(),
            'skewness': returns[ticker].skew()
        }
    compiled_data.append(ticker_data)

    returns_spy = returns['SPY']
    garch_result = run_garch_model(returns_spy)
    garch_result.update({
        'Start_Date': date_format(start_date),
        'End_Date': date_format(end_date)
    })
    garch_data.append(garch_result)


### Initial data and ticker variables
#Define the date range for data retrieval
end_date = date_format(pd.to_datetime('today').normalize()) #current date
start_date = date_format(pd.to_datetime(end_date) - timedelta(days=(5*365))) #5 years back from current date

# Define tickers for multi-asset analysis
ticker = 'SPY'

compiled_data = []  # Initialize an empty list to store compiled data
garch_data = []  # Initialize an empty list to store GARCH model data

compile_garch_forecasts(ticker, start_date, end_date, compiled_data, garch_data)

compiled_data = pd.DataFrame(compiled_data)
garch_data = pd.DataFrame(garch_data)
