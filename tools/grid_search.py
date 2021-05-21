import pandas as pd
from sklearn.metrics import mean_absolute_error
from typing import Dict

from config_field import ConfigField
from tools.wrapper_estimator import WrapperEstimator
from tools.splitter import Splitter


class GridSearch(object):
    _estimator = None
    _splitter: Splitter
    _error_params: Dict
    _params: Dict

    def __init__(self, config_field: ConfigField, estimator: WrapperEstimator, x_train: pd.DataFrame, y_train: pd.DataFrame):
        self._config_field = config_field
        self._estimator = estimator
        self._x = x_train
        self._y = y_train
        self._run()

    def _run(self) -> None:
        self._splitter = Splitter(self._x, self._y, self._config_field)

    @classmethod
    def _calc_grid(cls):
        cls._error_params = dict()
        for params in cls._estimator.param_grid:
            error = 0
            for pair in cls._splitter.train_test_pairs:
                cls._estimator.model.set_params(**params)
                cls._estimator.fit(pair['x_train'], pair['y_train'])
                y_pred = cls._estimator.predict_by_train_test(pair['y_train'], pair['x_test'])
                error += mean_absolute_error(pair['y_test'], y_pred)
            error /= cls._splitter.pair_number
            print(params, error)
            cls._error_params[error] = params

    @classmethod
    def _find_params(cls):
        error_min = min(cls._error_params.keys())
        cls._params = cls._error_params[error_min]
        print(cls._params)


class Splitter(object):

    _fold_samples_number = sett.forecast_days_number
    _r = 2

    def __init__(self, x: pd.DataFrame, y: pd.DataFrame):
        self.x = x
        self.y = y

        self.train_test_pairs = list()
        self.pair_number: int
        self._total_samples_number: int
        self._k_int: int
        self._k_frac: float

        self._calc()
        # self._calc_fold_number()
        # self._create_train_test_pair()
        # self._check_last_pair_existing()
        self._calc_pair_number()

    def _calc(self):
        total_samples_number = len(self.x.index)

        x_train = self.x.head(total_samples_number - self._fold_samples_number)
        y_train = self.y.head(total_samples_number - self._fold_samples_number)

        x_test = self.x.tail(self._fold_samples_number)
        y_test = self.y.tail(self._fold_samples_number)

        self.train_test_pairs.append({'x_train': x_train,
                                      'y_train': y_train,
                                      'x_test': x_test,
                                      'y_test': y_test})

    def _calc_fold_number(self):
        self._total_samples_number = len(self.x.index)
        k = self._total_samples_number / self._fold_samples_number
        self._k_int = int(k)
        self._k_frac = k - int(k)

    def _create_train_test_pair(self):
        for i in range(self._r, self._k_int + 1):
            samples_number = self._fold_samples_number * i
            x_k = self.x.head(samples_number)
            y_k = self.y.head(samples_number)

            x_train = x_k.head(samples_number - self._fold_samples_number)
            y_train = y_k.head(samples_number - self._fold_samples_number)

            x_test = x_k.tail(self._fold_samples_number)
            y_test = y_k.tail(self._fold_samples_number)

            self.train_test_pairs.append({'x_train': x_train,
                                          'y_train': y_train,
                                          'x_test': x_test,
                                          'y_test': y_test})

    def _check_last_pair_existing(self):
        if self._k_frac != 0:
            int_part = self._fold_samples_number * self._k_int
            frac_part = self._total_samples_number - int_part

            x_train = self.x.head(int_part)
            y_train = self.y.head(int_part)

            x_test = self.x.tail(frac_part)
            y_test = self.y.tail(frac_part)

            self.train_test_pairs.append({'x_train': x_train,
                                          'y_train': y_train,
                                          'x_test': x_test,
                                          'y_test': y_test})

    def _calc_pair_number(self):
        self.pair_number = len(self.train_test_pairs)
