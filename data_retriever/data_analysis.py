import datetime
from data_retriever.retriever import get_daily_historical
from utils.plot_factory import plot_historical

from utils.defaults import SYMBOLS
from utils.features import sma, ema, vwap


if __name__ == '__main__':

    # symbol = 'AAPL'
    # symbol = '^GSPC'
    for symbol in SYMBOLS.keys():
        start_date = datetime.datetime(2016, 1, 1)
        end_date = datetime.datetime(2020, 8, 31)

        # get OHLC data
        data = get_daily_historical(symbol, start_date, end_date)

        # compute features
        targets = data['Adj Close']

        # compute features
        data['sma_20'] = sma(targets, periods=20)
        data['sma_100'] = sma(targets, periods=100)
        data['ema_10'] = ema(targets, periods=10)
        data['ema_50'] = ema(targets, periods=50)
        data['vwap_20'] = vwap(price_data=targets, volume_data=data['Volume'], periods=20)
        data['vwap_100'] = vwap(price_data=targets, volume_data=data['Volume'], periods=100)

        for lag in [1, 7, 14, 28]:
            data[f"diff_{lag}"] = targets.diff(periods=lag)

        data = data.dropna()  # drop rows with at least 1 nans

        # plot data
        plot_historical(symbol, data, open_file=True)
