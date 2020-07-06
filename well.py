import numpy as np

import settings as sett

from pandas import Series, DataFrame, DatetimeIndex
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from typing import List

from tools.splitter import Splitter
from tools.window_feature_generator import WindowFeatureGenerator


class Well(object):

    _estimator = ElasticNet(alpha=10, l1_ratio=0.5, max_iter=10000)

    def __init__(self, well_name: str, df: DataFrame):
        self.name = well_name
        self.df = df

        self.df_with_new_features: DataFrame
        self.y_adap: Series
        self.y_fore: Series
        self.y_dev: Series
        self.MAE_adap: float
        self.MAE_fore: float

        self._add_features_to_df()
        self._split_divide_df()
        self._calc_adaptation()
        self._calc_forecast()
        self._calc_deviations()

    def _add_features_to_df(self):
        self.df_with_new_features = WindowFeatureGenerator.run(self.df)

    def _split_divide_df(self):
        df_train, df_test = Splitter.split_train_test(self.df_with_new_features)
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

    def _calc_deviations(self):
        self.y_dev = self._calc_relative_deviations(y_true=self._y_test, y_pred=self.y_fore)
        self.MAE_adap = mean_absolute_error(y_true=self._y_train, y_pred=self.y_adap)
        self.MAE_fore = mean_absolute_error(y_true=self._y_test, y_pred=self.y_fore)

    @staticmethod
    def _divide_x_y(df):
        x = df.drop(columns=sett.predicate)
        y = df[sett.predicate]
        return x, y

    @staticmethod
    def _create_series(y: List[float], dates: DatetimeIndex) -> Series:
        y = Series(y, dates)
        return y

    @staticmethod
    def _calc_relative_deviations(y_true: Series, y_pred: Series) -> Series:
        y_dev = []
        for i in y_true.index:
            y1 = y_true.loc[i]
            y2 = y_pred.loc[i]
            yd = abs(y1 - y2) / max(y1, y2) * 100
            y_dev.append(yd)
        y_dev = Series(y_dev, y_true.index)
        return y_dev
