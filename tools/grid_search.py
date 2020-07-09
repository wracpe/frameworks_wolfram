import settings as sett

from sklearn.metrics import mean_absolute_error
from typing import Dict

from tools.splitter import Splitter


class GridSearch(object):
    _estimator = None
    _splitter: Splitter
    _error_params: Dict
    _params: Dict

    @classmethod
    def run(cls, estimator, splitter: Splitter) -> Dict:
        cls._estimator = estimator
        cls._splitter = splitter

        cls._calc_grid()
        cls._find_params()
        return cls._params

    @classmethod
    def _calc_grid(cls):
        cls._error_params = dict()
        for params in cls._estimator.param_grid:

            cls._estimator.set_params(**params)
            error = 0
            i = 0
            for pair in cls._splitter.train_test_pairs:
                x_train, y_train = cls._divide_x_y(pair['train'])
                x_test, y_test = cls._divide_x_y(pair['test'])
                try:
                    cls._estimator.fit(x_train, y_train)
                    y_pred = cls._estimator.predict(x_test)
                    error += mean_absolute_error(y_test, y_pred)
                except:
                    continue
                i += 1
            error /= i
            cls._error_params[error] = params


    @classmethod
    def _find_params(cls):
        error_min = min(cls._error_params.keys())
        cls._params = cls._error_params[error_min]

    @staticmethod
    def _divide_x_y(df):
        x = df.drop(columns=sett.predicate)
        y = df[sett.predicate]
        return x, y
