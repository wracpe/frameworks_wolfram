import settings as sett

from sklearn.metrics import mean_absolute_error
from typing import Dict

from tools.estimator import Estimator
from tools.splitter import Splitter


class GridSearch(object):
    _estimator = None
    _splitter: Splitter
    _error_params: Dict
    _params: Dict

    @classmethod
    def run(cls, splitter: Splitter) -> Dict:
        cls._splitter = splitter

        cls._calc_grid()
        cls._find_params()
        return cls._params

    @classmethod
    def _calc_grid(cls):
        cls._error_params = dict()
        for params in Estimator.param_grid:
            error = 0
            for pair in cls._splitter.train_test_pairs:
                x_train, y_train = cls._divide_x_y(pair['train'])
                x_test, y_test = cls._divide_x_y(pair['test'])
                y_pred = Estimator.make_forecast(x_train, x_test, y_train, in_out='out', **params)
                error += mean_absolute_error(y_test, y_pred)
            error /= cls._splitter.pair_number
            print(params, error)
            cls._error_params[error] = params

    @classmethod
    def _find_params(cls):
        error_min = min(cls._error_params.keys())
        cls._params = cls._error_params[error_min]
        print(cls._params)

    @staticmethod
    def _divide_x_y(df):
        x = df.drop(columns=sett.predicate)
        y = df[sett.predicate]
        return x, y
