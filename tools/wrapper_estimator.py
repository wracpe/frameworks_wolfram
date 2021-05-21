import numpy as np
import pandas as pd
from itertools import product
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor

from config_field import ConfigField


class WrapperEstimator(object):

    def __init__(self, config_field: ConfigField, estimator_name: str):
        self._config_field = config_field
        self.estimator = _Estimator(estimator_name)

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        self.estimator.model.fit(x_train, y_train)

    def predict_by_test(self, x_test: pd.DataFrame):
        y_pred = self.estimator.model.predict(x_test)
        y_pred = pd.Series(y_pred, x_test.index)
        return y_pred

    def predict_by_train_test(self, y_train: pd.DataFrame, x_test: pd.DataFrame):
        y_pred = list()
        y_stat = y_train.to_list()
        prediction_length = len(x_test.index)

        for day in range(prediction_length):
            x = x_test.iloc[[day]]
            y = self.estimator.model.predict(x)[0]
            y_pred.append(y)
            y_stat.append(y)
            if day == prediction_length - 1:
                break

            for ws in self._config_field.window_sizes:
                parameter_name = self._config_field.predicate + f'_{ws}'
                window = y_stat[-ws:]
                x_test[f'{parameter_name}_mean'].iloc[day + 1] = np.mean(window)
                x_test[f'{parameter_name}_var'].iloc[day + 1] = np.var(window)
                for q in self._config_field.quantiles:
                    x_test[f'{parameter_name}_quantile_{q}'].iloc[day + 1] = np.quantile(window, q)

        y_pred = pd.Series(y_pred, x_test.index)
        return y_pred


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
        self.name = name
        self._run()

    def _run(self) -> None:
        self.model, self.param_dict = self._estimators[self.name]
        self.param_grid = []
        for params in product(self.param_dict.values()):
            x = dict(zip(self.param_dict.keys(), params))
            self.param_grid.append(x)
