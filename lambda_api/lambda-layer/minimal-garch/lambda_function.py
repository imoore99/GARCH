import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from arch import arch_model
import boto3
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

def lambda_handler(event, context):
    """
    AWS Lambda function to run daily GARCH volatility forecasting
    """
    try:
        # Initialize S3 client
        s3 = boto3.client('s3')
        bucket_name = 'garch-data'
        
        # File keys for both datasets
        compiled_data_key = 'compiled_data.csv'
        garch_data_key = 'garch_data.csv'
        
        # Step 1: Read existing data from S3
        existing_compiled_data = read_data_from_s3(s3, bucket_name, compiled_data_key)
        existing_garch_data = read_data_from_s3(s3, bucket_name, garch_data_key)
        
        # Step 2: Generate new data for today
        new_compiled_data, new_garch_data = generate_daily_data()
        
        # Step 3: Append new data to existing datasets
        updated_compiled_data = append_new_data(existing_compiled_data, new_compiled_data)
        updated_garch_data = append_new_data(existing_garch_data, new_garch_data)
        
        # Step 4: Save both updated datasets back to S3
        save_data_to_s3(s3, bucket_name, compiled_data_key, updated_compiled_data)
        save_data_to_s3(s3, bucket_name, garch_data_key, updated_garch_data)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Both datasets updated successfully',
                'forecast_date': new_garch_data['End_Date'],
                'volatility_forecast': float(new_garch_data['volatility_forecast']),
                'annualized_volatility': float(new_garch_data['annualized_volatility']),
                'mean_return': float(new_compiled_data['Mean_Return'])
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

def read_data_from_s3(s3, bucket_name, file_key):
    """
    Read existing data from S3 (works for both compiled_data and garch_data)
    """
    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content), index_col=0)
        return df
    except Exception as e:
        print(f"Error reading {file_key} from S3: {e}")
        # Return empty DataFrame with appropriate structure based on file type
        if 'compiled_data' in file_key:
            return pd.DataFrame(columns=['Ticker', 'Start_Date', 'End_Date', 'Mean_Return', 
                                       'Volatility', 'Max_Return', 'Min_Return', 'kurtosis', 'skewness'])
        else:  # garch_data
            return pd.DataFrame(columns=['Start_Date', 'End_Date', 'volatility_forecast', 'annualized_volatility'])

def generate_daily_data():
    """
    Generate both compiled market data and GARCH forecast for today using 5-year rolling window
    """
    # Calculate dates for 5-year rolling window
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=5*365)
    
    ticker = 'SPY'
    
    # Load market data
    returns = load_market_data(ticker, start_date, end_date)
    
    # Generate compiled market statistics
    compiled_data = {
        'Ticker': ticker,
        'Start_Date': start_date.strftime('%Y-%m-%d'),
        'End_Date': end_date.strftime('%Y-%m-%d'),
        'Mean_Return': returns[ticker].mean(),
        'Volatility': returns[ticker].std(),
        'Max_Return': returns[ticker].max(),
        'Min_Return': returns[ticker].min(),
        'kurtosis': returns[ticker].kurtosis(),
        'skewness': returns[ticker].skew()
    }
    
    # Run GARCH model
    garch_result = run_garch_model(returns[ticker])
    
    # Generate GARCH data
    garch_data = {
        'Start_Date': start_date.strftime('%Y-%m-%d'),
        'End_Date': end_date.strftime('%Y-%m-%d'),
        'volatility_forecast': garch_result['volatility_forecast'],
        'annualized_volatility': garch_result['annualized_volatility']
    }
    
    return compiled_data, garch_data

def load_market_data(ticker, start_date, end_date):
    """
    Load market data for specified ticker within a date range.
    Returns a DataFrame with returns.
    """
    # Load market data
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data = pd.Series(data['Close'], index=data.index)
    
    close = pd.DataFrame(data)
    returns = close.pct_change().dropna() * 100  # Convert to percentage
    return returns

def run_garch_model(returns_data):
    """
    Fit GARCH model and return volatility forecast
    """
    # Fit GARCH(1,1) model with Student's t-distribution
    model = arch_model(returns_data, vol='Garch', p=1, q=1, dist='t')
    fitted_model = model.fit(disp='off')
    
    # Get 1-day forecast
    single_day_forecast = fitted_model.forecast(horizon=1)
    base_daily_vol = np.sqrt(single_day_forecast.variance.iloc[-1, 0])
    
    return {
        'volatility_forecast': base_daily_vol,
        'annualized_volatility': base_daily_vol * np.sqrt(252),
        'date': returns_data.index[-1],
        'forecast_date': returns_data.index[-1] + pd.Timedelta(days=1)
    }

def append_new_data(existing_data, new_data):
    """
    Append new data to existing DataFrame (works for both datasets)
    """
    # Convert new data to DataFrame row
    new_row = pd.DataFrame([new_data])
    
    # Append to existing data
    updated_data = pd.concat([existing_data, new_row], ignore_index=True)
    
    return updated_data

def save_data_to_s3(s3, bucket_name, file_key, data):
    """
    Save updated data back to S3 (works for both datasets)
    """
    # Convert DataFrame to CSV string
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=True)
    
    # Upload to S3
    s3.put_object(
        Bucket=bucket_name,
        Key=file_key,
        Body=csv_buffer.getvalue(),
        ContentType='text/csv'
    )
    
    print(f"Updated data saved to s3://{bucket_name}/{file_key}")