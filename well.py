import settings as sett

from pandas import Series, DataFrame
from sklearn.metrics import mean_absolute_error

from tools.estimator import Estimator
from tools.grid_search import GridSearch
from tools.splitter import Splitter
from tools.window_feature_generator import WindowFeatureGenerator


class Well(object):

    def __init__(self, well_name: str, df: DataFrame):
        self.name = well_name
        self.df = df

        self.y_adap: Series
        self.y_pred: Series
        self.y_dev: Series
        self.MAE_adap: float
        self.MAE_fore: float

        self._add_features_to_df()
        self._make_forecast()
        self._calc_deviations()

    def _add_features_to_df(self):
        self.df_with_new_features = WindowFeatureGenerator.run(self.df)

    def _make_forecast(self):
        df_train, df_test = self._split_df(self.df_with_new_features)
        splitter = Splitter(df_train)
        if self.name == '68Р':
            params = {'alpha': 5, 'l1_ratio': 0.5}
        else:
            params = GridSearch.run(splitter)

        self._x_train, self._y_train = self._divide_x_y(df_train)
        self._x_test, self._y_test = self._divide_x_y(df_test)
        self.y_adap = Estimator.make_forecast(self._x_train, self._x_test, self._y_train, in_out='in', **params)
        self.y_pred = Estimator.make_forecast(self._x_train, self._x_test, self._y_train, in_out='out', **params)

    def _calc_deviations(self):
        self.y_dev = self._calc_relative_deviations(y_true=self._y_test, y_pred=self.y_pred)
        self.MAE_adap = mean_absolute_error(y_true=self._y_train, y_pred=self.y_adap)
        self.MAE_fore = mean_absolute_error(y_true=self._y_test, y_pred=self.y_pred)

    @staticmethod
    def _split_df(df):
        total_samples_number = len(df.index)
        df_train = df.head(total_samples_number - sett.forecast_days_number)
        df_test = df.tail(sett.forecast_days_number)
        return df_train, df_test

    @staticmethod
    def _divide_x_y(df):
        x = df.drop(columns=sett.predicate)
        y = df[sett.predicate]
        return x, y

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
