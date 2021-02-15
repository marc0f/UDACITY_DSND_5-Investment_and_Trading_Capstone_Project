import os

# thirds parts imports
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, load

# local imports
from utils.metrics import mean_absolute_percentage_error

NUM_CPUS = os.cpu_count() - 1


def build_model(num_cpus=NUM_CPUS):

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
    # cv = GridSearchCV(pipeline, param_grid=[srv_parameters, srv_parameters, srv_parameters, srv_parameters], verbose=2)
    # return cv
    return pipeline


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


def load_model(model_filepath):
    return load(model_filepath)
