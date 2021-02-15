import datetime
import logging

# thirds parts imports
import numpy as np

# local imports
from data_retriever.retriever import get_daily_historical
from utils.plot_factory import plot_historical
from utils.defaults import SYMBOLS
from utils.data_elaboration import clean_data, prepare_data, split_data
from utils.modelling import build_model, evaluate_model, save_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


np.random.seed(42)


if __name__ == '__main__':

    model_filepath = 'test.dump'
    prediction_horizons = [1, 7, 14, 28]  # steps of prediction in base resolution, i.e. days

    # sma 100
    features_extra_data_periods = 100

    test_len_days = 90  # days
    train_len_days = 2 * 365  # days

    symbol = list(SYMBOLS.keys())[0]

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

    X_train, X_test, Y_train, Y_test = split_data(samples, targets, test_size=test_len_days) #test_size=30, test_size=0.2

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, X_train, Y_train, prediction_horizons)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)
