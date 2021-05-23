import numpy as np
import pandas as pd
from itertools import product
from sklearn.linear_model import ElasticNet
from typing import Any, Dict, List
from xgboost import XGBRegressor

from config_field import ConfigField


class WrapperEstimator(object):

    def __init__(
            self,
            config_field: ConfigField,
            estimator_name: str,
    ):
        self._estimator = _Estimator(estimator_name)
        self._config_field = config_field

    def get_param_grid(self) -> List[Dict[str, Any]]:
        return self._estimator.param_grid

    def set_params(self, params: Dict[str, Any]) -> None:
        self._estimator.model.set_params(**params)

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        self._estimator.model.fit(x_train, y_train)

    def predict_train(self, x_train: pd.DataFrame) -> pd.DataFrame:
        ym_train = self._estimator.model.predict(x_train)
        ym_train = pd.DataFrame(ym_train, x_train.index)
        return ym_train

    def predict_test(self, y_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
        ym_test = []
        ym_stat = y_train.squeeze().to_list()
        for day in range(self._config_field.forecast_days_number):
            x = x_test.iloc[[day]]
            y = self._estimator.model.predict(x)[0]
            ym_test.append(y)
            ym_stat.append(y)
            if day == self._config_field.forecast_days_number - 1:
                break
            for ws in self._config_field.window_sizes:
                parameter_name = self._config_field.predicate + f'_{ws}'
                window = ym_stat[-ws:]
                x_test[f'{parameter_name}_mean'].iloc[day + 1] = np.mean(window)
                x_test[f'{parameter_name}_var'].iloc[day + 1] = np.var(window)
                for q in self._config_field.quantiles:
                    x_test[f'{parameter_name}_quantile_{q}'].iloc[day + 1] = np.quantile(window, q)
        ym_test = pd.DataFrame(ym_test, x_test.index)
        return ym_test


class _Estimator(object):

    _estimators = {
        'ela': (
            ElasticNet(random_state=1),
            {
                'alpha': np.arange(1, 12, 2),
                'l1_ratio': np.arange(0.2, 1, 0.2),
            },
        ),
        'xgb': (
            XGBRegressor(
                verbosity=0,
                booster='gbtree',
                random_state=1,
            ),
            {
                'n_estimators': np.arange(20, 60, 10),
                'max_depth': np.arange(5, 11, 1),
                'learning_rate': [0.5, 0.7],
                'colsample_bytree': [0.3, 0.7],
                'reg_alpha': [0.1, 1, 10],
                'reg_lambda': [0.1, 1, 10],
            },
        ),
    }

    def __init__(self, name: str):
        self.model, self._param_dict = self._estimators[name]
        self.param_grid = []
        for params in product(self._param_dict.values()):
            self.param_grid.append(dict(zip(self._param_dict.keys(), params)))
