import datetime
import logging

# thirds parts imports
import numpy as np

# local imports
from data_retriever.retriever import get_daily_historical
from utils.plot_factory import plot_historical
import utils.defaults as defaults
from utils.data_elaboration import clean_data, prepare_data, split_data
from utils.modelling import build_model, evaluate_model, save_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


np.random.seed(42)


if __name__ == '__main__':

    dataset_len = defaults.TEST_LEN_DAYS + defaults.TRAIN_LEN_DAYS + max(defaults.PREDICTION_HORIZONS) # days
    dataset_len += defaults.EXTRA_DATA_PERIODS

    end_date = datetime.datetime.now() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=dataset_len)

    # train a model for each asset
    for symbol in list(defaults.SYMBOLS.keys()):

        try:
            model_filepath = f"local_models/models_{symbol}.dump"

            # get OHLCV data
            data = get_daily_historical(symbol, start_date, end_date, min_length=dataset_len)
            data = clean_data(data)
            samples, targets = prepare_data(data, delays=defaults.PREDICTION_HORIZONS)

            X_train, X_test, Y_train, Y_test = split_data(samples, targets, test_size=defaults.TRAIN_LEN_DAYS)
            print('Building model...')
            model = build_model()

            print('Training model...')
            model.fit(X_train, Y_train)

            print('Evaluating model...')
            evaluate_model(model, X_test, Y_test, X_train, Y_train, defaults.PREDICTION_HORIZONS)

            print('Saving model...\n    MODEL: {}'.format(model_filepath))
            save_model(model, model_filepath)

        except Exception as err:
            print(f"Unhandled error for asset {symbol}. \n Error details: {err}. \n Continue.")
            continue
