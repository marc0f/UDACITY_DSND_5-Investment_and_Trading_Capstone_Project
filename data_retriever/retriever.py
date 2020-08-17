import os
import datetime

import pandas as pd
from yfinance import Ticker

from utils.defaults import DATA_DIR


def create_filename(symbol: str, interval: str):
    return os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")


def store_csv(filename: str, data: pd.DataFrame):
    data.to_csv(filename)

def load_csv(filename: str):
    return pd.read_csv(filename, header=0, index_col=0, parse_dates=True)


def get_daily_historical(symbol: str, start_date: datetime, end_date: datetime):
    """ smart function to retrieve ohlcv data in the daily interval, with both adjusted and not adjusted closing price.
      if a csv for the requested symbol already exists, read it and download only the data for the missing date ranges,
      otherwise -first run- retrieve full date range and store to csv """

    interval='1d'

    csv_filename = create_filename(symbol, interval)

    # check if exists
    if os.path.isfile(csv_filename):
        # if exists load csv and check interval. request data only for missing date range and store updated DataFrame
        data = load_csv(csv_filename)

        data_start_date = data.first_valid_index().to_pydatetime()
        if start_date < data_start_date:
            # requested interval starts before the stored data interval.
            previous_data = _get_historical(symbol, start_date, data_start_date, interval)
            if not previous_data.empty:
                data = data.append(previous_data)
                data.sort_index(inplace=True)
                store_csv(csv_filename, data)

        data_end_date = data.last_valid_index().to_pydatetime()
        if end_date > data_end_date:
            # requested interval ends after the stored data interval.
            following_data = _get_historical(symbol, data_end_date, end_date, interval)
            if not following_data.empty:
                data = data.append(following_data)
                data.sort_index(inplace=True)
                store_csv(csv_filename, data)

    else:
        # first download, request full range and store to csv
        data = _get_historical(symbol, start_date, end_date, interval)
        data.to_csv(csv_filename)

    return data[start_date:end_date]


def _get_historical(symbol: str, start_date: datetime, end_date: datetime, interval: str):
    """ returns ohlcv data in the specified interval, with both adjusted and not adjusted closing price.
      """

    # init symbol instance
    symbol_instance = Ticker(symbol)

    # request and returns tickers in interval
    return symbol_instance.history(start=start_date, end=end_date, interval=interval, auto_adjust=False)
