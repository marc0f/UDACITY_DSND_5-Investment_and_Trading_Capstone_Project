import datetime

from yfinance import Ticker


def get_historical(symbol: str, start_date: datetime, end_date: datetime):
    """ return ohlcv data in the daily interval, with both adjusted and not adjustec closing price. """

    # init symbol instance
    symbol_instance = Ticker(symbol)

    # request and returns tickers in interval
    return symbol_instance.history(start=start_date, end=end_date, interval='1d', auto_adjust=False)
