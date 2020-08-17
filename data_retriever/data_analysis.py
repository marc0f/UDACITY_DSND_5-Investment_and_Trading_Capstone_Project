import datetime
from data_retriever.retriever import get_daily_historical
from utils.plot_factory import plot_historical

from utils.defaults import DEFAULT_SYMBOLS


if __name__ == '__main__':

    # symbol = 'AAPL'
    # symbol = '^GSPC'
    for symbol in DEFAULT_SYMBOLS.keys():
        start_date = datetime.datetime(2016, 1, 1)
        end_date = datetime.datetime(2020, 8, 31)

        # get OHLC data
        data = get_daily_historical(symbol, start_date, end_date)

        # plot data
        plot_historical(symbol, data)
