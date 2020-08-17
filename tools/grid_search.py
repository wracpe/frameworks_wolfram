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
    def run(cls, splitter: Splitter, estimator: Estimator) -> Dict:
        cls._splitter = splitter
        cls._estimator = estimator

        cls._calc_grid()
        cls._find_params()
        return cls._params

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
