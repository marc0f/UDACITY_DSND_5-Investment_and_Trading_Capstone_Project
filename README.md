# Data Science Data Scientist Nanodegree. Capstore Project: Investment and Trading

## Capstore Report

The complete report is available in the folder [report](report/REPORT.md)


## Get started

At first have a look of the retrieved data by running:

    python data_retriever/data_analysis.py


Create update models by running:

    python models/train_regressor.py

The updated model for each asset is stored in the folder _models/local_models_. To use in production a specifi model,
copy/move it into the folder  _models/production_models_


## TODO

- dashboard: display last N-days of data and provides prediction points (1, 7, 14, 28 days) (vertical stuck of plots).
- dashboard: top view provides a quick oerview of available symbols and reports estimated pct change at different
horizon times, together with color box: light to dark green for increasing positive returns, light to dark red for increasing negative returns.


## References

Project main reference: https://docs.google.com/document/d/1ycGeb1QYKATG6jvz74SAMqxrlek9Ed4RYrzWNhWS-0Q/pub

Available API for financial:
 - (broken) Yahoo! Finance: (broken)[]https://pypi.org/project/yahoo-finance/] (works)[https://github.com/ranaroussi/yfinance]
 - Bloomberg: https://www.bloomberg.com/professional/support/api-library/ [package doc](https://bloomberg.github.io/blpapi-docs/python/3.13/)
 - (pay for use) Quandl: https://docs.quandl.com/docs/python-installation [Stock prices doc](https://www.quandl.com/data/EOD-End-of-Day-US-Stock-Prices/documentation)
