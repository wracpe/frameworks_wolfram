import numpy as np

import settings as sett

from pandas import Series, DataFrame, DatetimeIndex
from sklearn.linear_model import ElasticNet
from typing import List

from tools.splitter import Splitter
from tools.window_feature_generator import WindowFeatureGenerator


class Well(object):

    y_adap: Series
    y_fore: Series
    _estimator = ElasticNet(alpha=5, l1_ratio=0.7, max_iter=10000)

    def __init__(self, well_name: str, df: DataFrame):
        self.well_name = well_name
        self.df = WindowFeatureGenerator.run(df)

        self._split_divide_df()
        self._calc_adaptation()
        self._calc_forecast()

    def _split_divide_df(self):
        df_train, df_test = Splitter.split_train_test(self.df)
        self._x_train, self._y_train = self._divide_x_y(df_train)
        self._x_test, self._y_test = self._divide_x_y(df_test)

    def _calc_adaptation(self):
        self._estimator.fit(self._x_train, self._y_train)
        y_adap = self._estimator.predict(self._x_train)
        self.y_adap = self._create_series(y_adap, self._x_train.index)

    def _calc_forecast(self):
        y_fore = []
        y_statistics = self._y_train.to_list()
        for i in range(sett.forecast_days_number):
            x = self._x_test.iloc[[i]]
            y = self._estimator.predict(x)[0]
            y_fore.append(y)
            y_statistics.append(y)
            if i == sett.forecast_days_number - 1:
                break
            for ws in sett.window_sizes:
                parameter_name = sett.predicate + f'_{ws}'
                window = y_statistics[-ws:]
                self._x_test[f'{parameter_name}_median'].iloc[i + 1] = np.median(window)
                self._x_test[f'{parameter_name}_var'].iloc[i + 1] = np.var(window)
                for q in sett.quantiles:
                    self._x_test[f'{parameter_name}_quantile_{q}'].iloc[i + 1] = np.quantile(window, q)
        self.y_fore = self._create_series(y_fore, self._x_test.index)

    @staticmethod
    def _divide_x_y(df):
        x = df.drop(columns=sett.predicate)
        y = df[sett.predicate]
        return x, y

    @staticmethod
    def _create_series(y: List[float], dates: DatetimeIndex) -> Series:
        y = Series(y, dates)
        return y
