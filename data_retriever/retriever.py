import os
import datetime
import logging

import pandas as pd
from yfinance import Ticker

from utils.defaults import DATA_DIR, interval_to_seconds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


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

    logger.info(f"Data request: {symbol} from {start_date} to {end_date}")

    interval='1d'
    interval_in_seconds = interval_to_seconds(interval)

    csv_filename = create_filename(symbol, interval)

    # check if exists
    if os.path.isfile(csv_filename):
        logger.info(f"Data already in {csv_filename}. Check for missing dates to fill...")

        # if exists load csv and check interval. request data only for missing date range and store updated DataFrame
        data = load_csv(csv_filename)

        # TMP: check for duplicated
        if data.index.duplicated().any():
            logger.warning("Found duplicated indexes, cleaning data and re-store.")
            mask_no_duplicates = ~data.index.duplicated(keep='first')
            data = data[mask_no_duplicates]
            store_csv(csv_filename, data)

        data_start_date = data.first_valid_index().to_pydatetime() - pd.Timedelta(seconds=interval_in_seconds)
        if start_date < data_start_date:

            logger.info(f"Partial range requested to provider: {start_date} to {data_start_date}")

            # requested interval starts before the stored data interval.
            previous_data = _get_historical(symbol, start_date, data_start_date, interval)
            if previous_data.index.isin(data.index).all():
                logger.info(f"Received dates already present.")

            elif not previous_data.empty:
                data = data.append(previous_data)
                data.sort_index(inplace=True)
                store_csv(csv_filename, data)
                logger.info(f"Data received and csv updated")

            else:
                logger.info(f"No data received.")

        data_end_date = data.last_valid_index().to_pydatetime() + pd.Timedelta(seconds=interval_in_seconds)
        if end_date > data_end_date:

            logger.info(f"Partial range requested to provider: {data_end_date} to {end_date}")

            # requested interval ends after the stored data interval.
            following_data = _get_historical(symbol, data_end_date, end_date, interval)
            if following_data.index.isin(data.index).all():
                logger.info(f"Received dates already present.")

            elif not following_data.empty:
                data = data.append(following_data)
                data.sort_index(inplace=True)
                store_csv(csv_filename, data)
                logger.info(f"Data received and csv updated")

            else:
                logger.info(f"No data received.")


    else:
        # first download, request full range and store to csv
        logger.info(f"Request full range to provider.")
        data = _get_historical(symbol, start_date, end_date, interval)
        data.to_csv(csv_filename)
        logger.info(f"Data received and stored in {csv_filename}")

    return data[start_date:end_date]


def _get_historical(symbol: str, start_date: datetime, end_date: datetime, interval: str):
    """ returns ohlcv data in the specified interval, with both adjusted and not adjusted closing price.
      """

    # init symbol instance
    symbol_instance = Ticker(symbol)

    # request and returns tickers in interval
    return symbol_instance.history(start=start_date, end=end_date, interval=interval, auto_adjust=False)
