import os
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump
from sklearn.utils import check_array
import logging

# local imports
from data_retriever.retriever import get_daily_historical
from utils.plot_factory import plot_historical
from utils.defaults import DEFAULT_SYMBOLS
from utils.features import sma, ema, vwap


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

num_cpus = os.cpu_count() - 1
np.random.seed(42)


def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()


def mean_absolute_percentage_error(y_true, y_pred):
    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


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


def data_generator(data, labels=None, lookback=0, delays=[0], indexes=None, shuffle=False, batch_size=128, step=1, non_stop=False):
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

    if isinstance(delays, (int, float)):
        delays = [delays]

    max_delay = max(delays)
    len_delay = len(delays)

    data = validate_input_data(data)
    labels = validate_input_data(labels)

    min_index = 0
    support_index = None  # use to get the actual index, allow to work with continues range
    if indexes is None:
        # use all data directly
        max_index = len(data) - max_delay - 1
        support_index = np.asanyarray(range(min_index, len(data)))

    else:
        # min and max index go from 0 to len(indexes), create support vector to max i -> indexes[i] -> data[indexes[i]]
        max_index = len(indexes) - max_delay - 1
        support_index = indexes

    i = min_index + lookback

    if batch_size == 0:
        batch_size = max_index + 1

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

    # targets = np.zeros((len(rows), len_delay))
    targets = np.zeros((len(rows), len_delay))

    for j, row in enumerate(rows):
        if lookback == 0:
            indices = [rows[j]]

        else:
            indices = range(rows[j] - lookback, rows[j], step)

        samples[j] = data[support_index[indices]]

        for i, _delay in enumerate(delays):
            targets[j, i] = labels[support_index[rows[j] + _delay]]

    return samples, targets


def prepare_data(data, delays=None):
    """ target is adjusted close"""

    targets = data['Adj Close']
    # samples = data.drop(columns='Adj Close')
    samples = data

    # compute features
    samples['sma_20'] = sma(targets, periods=20)
    samples['sma_100'] = sma(targets, periods=100)
    samples['ema_10'] = ema(targets, periods=10)
    samples['ema_50'] = ema(targets, periods=50)
    samples['vwap_20'] = vwap(price_data=targets, volume_data=samples['Volume'], periods=20)
    samples['vwap_100'] = vwap(price_data=targets, volume_data=samples['Volume'], periods=100)

    for lag in [1, 7, 14, 28]:
        samples[f"diff_{lag}"] = targets.diff(periods=lag)

    samples = samples.dropna()  # drop rows with at least 1 nans
    targets = targets[samples.index[0]:samples.index[-1]]

    samples, targets = data_generator(samples, targets, delays=delays, batch_size=0)

    return samples, targets


def build_model():

    # compose the processing pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        # ('scaler', Normalizer()),
        ('regres', MultiOutputRegressor(SVR(), n_jobs=num_cpus))
        # ('regres', MultiOutputRegressor(LinearRegression(), n_jobs=num_cpus))
        # ('regres', SVR())
    ])

    # # full params
    # parameters = {
    #     'regres__estimator__kernel': [10, 50, 100, 200],
    #     'regres__estimator__min_samples_split': [2, 3, 4]
    # }

    # reduced params
    # best_parameters = {
    #     'cls__estimator__kernel': ['linear', 'rgf'],
    #     'tfidf__use_idf': (True, False),
    #     'vect__max_df': 0.5,
    #     'vect__max_features': 10000,
    #     'vect__ngram_range': (1, 2)}

    srv_parameters = {
        'regres__estimator__C': np.arange(0.2, 2, step=0.2),
        # 'regres__estimator__C': np.arange(0.2, 2, step=0.5),
        # 'regres__estimator__cache_size': 200,
        # 'regres__estimator__coef0': 0.0,
        # 'regres__estimator__degree': 3,
        'regres__estimator__epsilon': np.arange(0.02, 0.2, step=0.02),
        # 'regres__estimator__epsilon': np.arange(0.02, 0.2, step=0.1),
        # 'regres__estimator__gamma': 'scale',
        'regres__estimator__kernel': ['linear', 'rbf'],
        # 'regres__estimator__max_iter': -1,
        # 'regres__estimator__shrinking': True,
        # 'regres__estimator__tol': 0.001
    }

    # instantiate search grid
    cv = GridSearchCV(pipeline, param_grid=[srv_parameters, srv_parameters, srv_parameters, srv_parameters], verbose=2)
    return cv
    # return pipeline


def evaluate_model(model, X_test, Y_test, X_train, Y_train, category_names):

    # use model to predict output given the test data
    Y_pred = model.predict(X_test)
    Y_pred_train = model.predict(X_train)

    # convert prediction and expected outputs into dataframes
    y_pred_df = pd.DataFrame(Y_pred)
    y_pred_df.columns = category_names
    y_test_df = pd.DataFrame(Y_test)
    y_test_df.columns = category_names
    ##
    y_pred_train_df = pd.DataFrame(Y_pred_train)
    y_pred_train_df.columns = category_names
    y_train_df = pd.DataFrame(Y_train)
    y_train_df.columns = category_names

    # get reports of the performance (accuracy, f1-score, precision, recall) for each category
    # reports = dict()
    print("Lags:\tMSE\t\t\t\t\tMAE\t\t\t\t\tMAPE")
    for col in category_names:
        mse = mean_squared_error(y_test_df[col], y_pred_df[col])
        mae = mean_absolute_error(y_test_df[col], y_pred_df[col])
        mape = mean_absolute_percentage_error(y_test_df[col], y_pred_df[col])
        print(f"{col}\t\t{mse}\t{mae}\t{mape}")

        # model performance
        mse_train = mean_squared_error(y_train_df[col], y_pred_train_df[col])
        mae_train = mean_absolute_error(y_train_df[col], y_pred_train_df[col])
        mape_train = mean_absolute_percentage_error(y_train_df[col], y_pred_train_df[col])
        print(f"({col}\t\t{mse_train}\t{mae_train}\t{mape_train})")

    # print best params when search grid is performed
    # model._final_estimator.estimators_[1].get_params()

    if isinstance(model, GridSearchCV):
        print("Best params:")
        print(model.best_params_)


def save_model(model, model_filepath):
    dump(model, model_filepath)


if __name__ == '__main__':

    model_filepath = 'test.dump'
    prediction_horizons = [1, 7, 14, 28]  # steps of prediction in base resolution, i.e. days

    # sma 100
    features_extra_data_periods = 100

    test_len_days = 90  # days
    train_len_days = 2 * 365  # months

    symbol = list(DEFAULT_SYMBOLS.keys())[0]

    dataset_len = test_len_days + train_len_days + max(prediction_horizons) # days
    dataset_len += features_extra_data_periods

    end_date = datetime.datetime.now() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=dataset_len)
    # start_date = datetime.datetime(2016, 1, 1)
    # dataset_len = 0

    # get OHLCV data
    data = get_daily_historical(symbol, start_date, end_date, min_length=dataset_len)
    data = clean_data(data)
    samples, targets = prepare_data(data, delays=prediction_horizons)

    X_train, X_test, Y_train, Y_test = train_test_split(samples, targets, test_size=test_len_days) #test_size=30, test_size=0.2

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, X_train, Y_train, prediction_horizons)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)
