import numpy as np

import settings as sett

from itertools import product
from pandas import Series, DataFrame
from sklearn.linear_model import LinearRegression, ElasticNet
from xgboost import XGBRegressor


class Estimator(object):

    def __init__(self, model_name: str):
        if model_name == 'lin_reg':
            self.model = LinearRegression()

        if model_name == 'elastic_net':
            self.model = ElasticNet(alpha=1,
                                    l1_ratio=0.8,
                                    precompute=True,
                                    tol=1e-3,
                                    random_state=1)

            alpha = np.arange(0.2, 1.2, 0.2)
            l1_ratio = np.arange(0.2, 1, 0.2)
            self.param_grid = [{'alpha': x[0], 'l1_ratio': x[1]} for x in product(alpha, l1_ratio)]

        if model_name == 'xgb':
            self.model = XGBRegressor(n_estimators=30,
                                      max_depth=10,
                                      learning_rate=0.7,
                                      verbosity=0,
                                      objective='reg:squarederror',
                                      booster='gbtree',
                                      n_jobs=8,
                                      colsample_bytree=0.3,
                                      reg_alpha=0.1,
                                      reg_lambda=0.1,
                                      random_state=1)

            # n_estimators = np.arange(10, 60, 10)
            max_depth = np.arange(3, 11)
            colsample_bytree = np.arange(0.1, 1, 0.1)
            self.param_grid = [{'max_depth': x[0], 'colsample_bytree': x[1]} for x in product(max_depth, colsample_bytree)]

    def fit(self, x_train: DataFrame, y_train: Series):
        self.model.fit(x_train, y_train)

    def predict_by_test(self, x_test: DataFrame):
        y_pred = self.model.predict(x_test)
        y_pred = Series(y_pred, x_test.index)
        return y_pred

    def predict_by_train_test(self, y_train: Series, x_test: DataFrame):
        y_pred = list()
        y_stat = y_train.to_list()
        prediction_length = len(x_test.index)

        for day in range(prediction_length):
            x = x_test.iloc[[day]]
            y = self.model.predict(x)[0]
            y_pred.append(y)
            y_stat.append(y)
            if day == prediction_length - 1:
                break

            for ws in sett.window_sizes:
                parameter_name = sett.predicate + f'_{ws}'
                window = y_stat[-ws:]
                x_test[f'{parameter_name}_mean'].iloc[day + 1] = np.mean(window)
                x_test[f'{parameter_name}_var'].iloc[day + 1] = np.var(window)
                for q in sett.quantiles:
                    x_test[f'{parameter_name}_quantile_{q}'].iloc[day + 1] = np.quantile(window, q)

        y_pred = Series(y_pred, x_test.index)
        return y_pred
