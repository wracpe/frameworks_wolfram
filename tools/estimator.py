import numpy as np

import settings as sett

from itertools import product
from pandas import Series, DataFrame
from sklearn.linear_model import ElasticNet
from typing import Tuple
from xgboost import XGBRegressor


def get_param_grid(model_name: str) -> Tuple:
    model = None
    param_grid = None

    if model_name == 'eln':
        model = ElasticNet()
        alpha_ = np.arange(0, 11, 1)
        l1_ratio_ = np.arange(0, 1.1, 0.1)
        param_grid = [{'alpha': x[0], 'l1_ratio': x[1]} for x in product(alpha_, l1_ratio_)]

    if model_name == 'xgb':
        model = XGBRegressor(booster='gbtree', objective='reg:squarederror')
        eta_ = np.arange(0.7, 1.1, 0.1)
        lambda_ = np.arange(8, 11, 1)
        alpha_ = np.arange(0.7, 1.1, 0.1)
        param_grid = [{'eta': x[0], 'lambda': x[1], 'alpha': x[2]} for x in product(eta_, lambda_, alpha_)]

    return model, param_grid


class Estimator(object):

    model, param_grid = get_param_grid(model_name='eln')
    _x_train: DataFrame
    _x_test: DataFrame
    _y_train: Series
    _y_pred: Series

    @classmethod
    def make_forecast(cls, x_train, x_test, y_train, in_out: str = 'out', **params):
        cls._x_train = x_train
        cls._x_test = x_test
        cls._y_train = y_train

        cls.model.set_params(**params)
        cls.model.fit(cls._x_train, cls._y_train)
        if in_out == 'in':
            cls._predict_in_samples()
        if in_out == 'out':
            cls._predict_out_samples()
        return cls._y_pred

    @classmethod
    def _predict_in_samples(cls):
        y_pred = cls.model.predict(cls._x_train)
        cls._y_pred = Series(y_pred, cls._x_train.index)

    @classmethod
    def _predict_out_samples(cls):
        y_pred = list()
        y_stat = cls._y_train.to_list()
        prediction_length = len(cls._x_test.index)

        for day in range(prediction_length):
            x = cls._x_test.iloc[[day]]
            y = cls.model.predict(x)[0]
            y_pred.append(y)
            y_stat.append(y)
            if day == prediction_length - 1:
                break

            for ws in sett.window_sizes:
                parameter_name = sett.predicate + f'_{ws}'
                window = y_stat[-ws:]
                cls._x_test[f'{parameter_name}_median'].iloc[day + 1] = np.median(window)
                cls._x_test[f'{parameter_name}_var'].iloc[day + 1] = np.var(window)
                for q in sett.quantiles:
                    cls._x_test[f'{parameter_name}_quantile_{q}'].iloc[day + 1] = np.quantile(window, q)

        cls._y_pred = Series(y_pred, cls._x_test.index)
