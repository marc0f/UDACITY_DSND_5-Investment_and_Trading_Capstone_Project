import json
import datetime
import numpy as np
import pandas as pd
import logging


from flask import Flask
from flask import render_template, request, jsonify
import plotly

from data_retriever.retriever import get_daily_historical
from utils.plot_factory import plot_historical_with_predicted_data
from utils.defaults import DEFAULT_SYMBOLS
from models.train_regressor import prepare_data, clean_data, load_model


app = Flask(__name__)
logger = logging.getLogger(__name__)

# index webpage displays cool visuals
@app.route('/')
@app.route('/index')
def index():

    figures = list()

    # start_date = datetime.datetime(2016, 1, 1)
    # end_date = datetime.datetime(2020, 8, 31)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=180)

    prediction_horizons = [1, 7, 14, 28]

    regression_result_1d = dict()
    regression_result_7d = dict()
    regression_result_14d = dict()
    regression_result_28d = dict()

    for symbol in DEFAULT_SYMBOLS.keys():

        logger.info(f"Performing regression for symbol: {symbol}")
        model = load_model(f"../models/production_models/models_{symbol}.dump")

        # get OHLCV data
        data = get_daily_historical(symbol, start_date, end_date)

        data = clean_data(data)
        regression_input, _ = prepare_data(data.copy(), diffs=prediction_horizons)
        regression_input = regression_input[-1, :].reshape(1, -1)

        # compute predictions
        regression_outputs = model.predict(regression_input)[-1]

        current_price = data['Adj Close'].iloc[-1]
        regression_pct_returns = 100 * (regression_outputs - current_price) / current_price
        regression_pct_returns = np.round(regression_pct_returns, decimals=2)

        regression_result_1d.update({symbol: regression_pct_returns[0]})
        regression_result_7d.update({symbol: regression_pct_returns[1]})
        regression_result_14d.update({symbol: regression_pct_returns[2]})
        regression_result_28d.update({symbol: regression_pct_returns[3]})

        # create dataframe with predicted data
        predicted_data = pd.DataFrame(index=[data.index[-1]],
                                      data=[[data.iloc[-1]['Adj Close']]*4],
                                      columns=['Adj Close - 1d_prediction', 'Adj Close - 7d_prediction', 'Adj Close - 14d_prediction', 'Adj Close - 28d_prediction'])
        # append 1day prediction
        predicted_data = predicted_data.append(pd.DataFrame(index=[predicted_data.index[0] + pd.Timedelta(days=1)],
                                                            data=[regression_outputs[0]],
                                                            columns=['Adj Close - 1d_prediction']))
        # append 7day prediction
        predicted_data = predicted_data.append(pd.DataFrame(index=[predicted_data.index[0] + pd.Timedelta(days=7)],
                                                            data=[regression_outputs[1]],
                                                            columns=['Adj Close - 7d_prediction']))
        # append 14day prediction
        predicted_data = predicted_data.append(pd.DataFrame(index=[predicted_data.index[0] + pd.Timedelta(days=14)],
                                                            data=[regression_outputs[2]],
                                                            columns=['Adj Close - 14d_prediction']))
        # append 28day prediction
        predicted_data = predicted_data.append(pd.DataFrame(index=[predicted_data.index[0] + pd.Timedelta(days=28)],
                                                            data=[regression_outputs[3]],
                                                            columns=['Adj Close - 28d_prediction']))

        # plot data
        fig = plot_historical_with_predicted_data(symbol, data, predicted_data, return_fig=True)
        figures.append(fig)

    # encode plotly graphs in JSON
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]
    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html',
                           ids=ids,
                           figuresJSON=figuresJSON,
                           regression_result_1d=regression_result_1d,
                           regression_result_7d=regression_result_7d,
                           regression_result_14d=regression_result_14d,
                           regression_result_28d=regression_result_28d)


def main():
    app.run(host='0.0.0.0', port=8080, debug=True)


if __name__ == '__main__':
    main()