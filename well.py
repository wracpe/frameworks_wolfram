import settings as sett

from pandas import Series, DataFrame
from sklearn.metrics import mean_absolute_error
from typing import Dict

from tools.estimator import Estimator

import matplotlib.pyplot as plt

from tools.grid_search import GridSearch
from tools.splitter import Splitter


class Well(object):

    def __init__(self, well_name: str, data: Dict):
        self.name = well_name
        self.data = data

        self.y_adap: Series
        self.y_pred: Series
        self.y_dev: Series
        self.MAE_adap: float
        self.MAE_fore: float

        self._make_forecast()
        self._calc_deviations()

    def _make_forecast(self):
        estimator = Estimator(model_name='xgb')

        # splitter = Splitter(self.data['x_train'], self.data['y_train'])
        # params = GridSearch.run(splitter, estimator)
        # estimator.model.set_params(**params)
        estimator.fit(self.data['x_train'], self.data['y_train'])

        self.y_adap = estimator.predict_by_test(self.data['x_train'])
        self.y_pred = estimator.predict_by_train_test(self.data['y_train'], self.data['x_test'])

        # self.data['y_train'].plot()
        # self.y_adap.plot()
        # self.data['y_test'].plot()
        # self.y_pred.plot()
        # plt.show()

    def _calc_deviations(self):
        start = self.data['start_row'][sett.predicate].tolist()[0]

        fact = list()
        model = list()
        y_fact = start
        y_model = start
        for i in self.data['y_train'].index:
            y_fact = y_fact * (1 + self.data['y_train'].loc[i])
            y_model = y_fact * (1 + self.y_adap.loc[i])
            fact.append(y_fact)
            model.append(y_model)
        self.data['y_train'] = Series(fact, self.data['y_train'].index)
        self.y_adap = Series(model, self.data['y_train'].index)

        fact = list()
        model = list()
        y_fact = self.data['y_train'].iloc[-1]
        y_model = y_fact
        for i in self.data['y_test'].index:
            y_fact = y_fact * (1 + self.data['y_test'].loc[i])
            y_model = y_model * (1 + self.y_pred.loc[i])
            fact.append(y_fact)
            model.append(y_model)
        self.data['y_test'] = Series(fact, self.data['y_test'].index)
        self.y_pred = Series(model, self.data['y_test'].index)

        self.y_dev = self._calc_relative_deviations(y_true=self.data['y_test'], y_pred=self.y_pred)
        self.MAE_adap = mean_absolute_error(y_true=self.data['y_train'], y_pred=self.y_adap)
        self.MAE_fore = mean_absolute_error(y_true=self.data['y_test'], y_pred=self.y_pred)

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
