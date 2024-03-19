import yfinance as yf

def fetch_data(ticker, start_date, end_date):
    """
    Fetch historical market data for a given ticker between start_date and end_date.

    Parameters:
    ticker (str): The ticker symbol of the stock or commodity.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    pandas.DataFrame: Historical data for the ticker.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data
