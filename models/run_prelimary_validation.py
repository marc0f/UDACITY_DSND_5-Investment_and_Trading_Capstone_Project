import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# local imports
from data_retriever.retriever import get_daily_historical
from utils.plot_factory import plot_historical
from utils.defaults import DEFAULT_SYMBOLS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()


def clean_data(data):
    """ remove nans or constant values columns, or drop equals columns"""

    for col_name in list(data.columns):
        if is_unique(data[col_name]):
            logger.info(f"Column {col_name} has unique values..removed.")
            data.drop(columns=col_name, inplace=True)

    return data


def validate_input_data(data):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.values

    elif isinstance(data, np.ndarray):
        return data

    else:
        raise ValueError(f"Input data must be pd.DataFrame, pd.Series or np.ndarray. Received type: {type(data)}")


def data_generator(data, labels=None, lookback=0, delay=0, indexes=None, shuffle=False, batch_size=128, step=1, non_stop=False):
    """ subsample data and create delayed data
    inputs:
        data: dataframe to use as samples and, conditionally, as targets
        labels: if present use this as targets
        lookback: points in the past to consider in each sample
        delay: distance in the future of the target respect the sample
        indexes: list of index to select must be discontinues, as results of kfold, thus create a index-of-indexes to iterate over
                a continue range
        shuffle:
        batch_size:
        step: take a valid samples every n step
        non_stop: if True the data_generator continues to generate. If true, you need to set steps_per_epoch in fit params

        # Doc:
        data — The original array of floating-point data, which you normalized in listing
        lookback — How many timesteps back the input data should go.
        delay — How many timesteps in the future the target should be.
        min_index and max_index — Indices in the data array that delimit which timesteps to draw from.
                This is useful for keeping a segment of the data for validation and another for testing.
        shuffle — Whether to shuffle the samples or draw them in chronological order.
        batch_size — The number of samples per batch.
        step — The period, in timesteps, at which you sample data. You’ll set it to 6 in order to draw one data point every hour

    """

    data = validate_input_data(data)
    labels = validate_input_data(labels)

    min_index = 0
    support_index = None  # use to get the actual index, allow to work with continues range
    if indexes is None:
        # use all data directly
        max_index = len(data) - delay - 1
        support_index = np.asanyarray(range(min_index, len(data)))

    else:
        # min and max index go from 0 to len(indexes), create support vector to max i -> indexes[i] -> data[indexes[i]]
        max_index = len(indexes) - delay - 1
        support_index = indexes

    i = min_index + lookback

    if batch_size == 0:
        batch_size = max_index + 1

    # while True:
    #     if shuffle:
    #         rows = np.random.randint(
    #             min_index + lookback, max_index, size=batch_size)
    #     else:
    #         if i + batch_size >= max_index:
    #             i = min_index + lookback  # reset index
    #
    #         rows = np.arange(i, min(i + batch_size, max_index + 1))
    #         i += len(rows)
    #     if lookback == 0:
    #         samples = np.zeros((len(rows),
    #                             data.shape[-1]))
    #
    #     else:
    #         samples = np.zeros((len(rows),
    #                             lookback // step,
    #                             data.shape[-1]))
    #
    #     targets = np.zeros((len(rows),))
    #     for j, row in enumerate(rows):
    #         if lookback == 0:
    #             indices = [rows[j]]
    #
    #         else:
    #             indices = range(rows[j] - lookback, rows[j], step)
    #
    #         samples[j] = data[support_index[indices]]
    #         targets[j] = labels[support_index[rows[j] + delay]]
    #     yield samples, targets

    if shuffle:
        rows = np.random.randint(
            min_index + lookback, max_index, size=batch_size)
    else:
        if i + batch_size >= max_index:
            i = min_index + lookback  # reset index

        rows = np.arange(i, min(i + batch_size, max_index + 1))
        i += len(rows)

    if lookback == 0:
        samples = np.zeros((len(rows),
                            data.shape[-1]))

    else:
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))

    targets = np.zeros((len(rows),))
    for j, row in enumerate(rows):
        if lookback == 0:
            indices = [rows[j]]

        else:
            indices = range(rows[j] - lookback, rows[j], step)

        samples[j] = data[support_index[indices]]
        targets[j] = labels[support_index[rows[j] + delay]]
    return samples, targets


def prepare_data(data, delay=0):
    """ target is adjusted close"""

    targets = data['Adj Close']
    samples = data.drop(columns='Adj Close')

    samples, targets = data_generator(samples, targets, delay=delay, batch_size=0)

    return samples, targets


if __name__ == '__main__':

    prediction_horizon = 7  # steps of prediction in base resolution, i.e. days

    symbol = list(DEFAULT_SYMBOLS.keys())[0]

    start_date = datetime.datetime(2016, 1, 1)
    end_date = datetime.datetime(2020, 8, 31)

    # get OHLC data
    data = get_daily_historical(symbol, start_date, end_date)
    data = clean_data(data)
    samples, targets = prepare_data(data, delay=prediction_horizon)

    # plot data
    # plot_historical(symbol, data)

    X_train, X_test, Y_train, Y_test = train_test_split(samples, targets, test_size=0.2)
    print()