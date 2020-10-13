import pandas as pd

import settings as sett

from pandas import Series
from typing import Dict, List

from tools.estimator import Estimator
from well import Well

import matplotlib.pyplot as plt


class Project(object):

    def __init__(self, well_data: Dict[str, Dict]):
        self.well_data = well_data
        self.wells: List[Well]
        self.y_dev: Series

        self._calc_project()

    def _calc_project(self):
        x = list()
        y = list()
        for well_name, data in self.well_data.items():
            x.append(data['x_train'])
            y.append(data['y_train'])
        x_train = pd.concat(objs=x, ignore_index=True)
        y_train = pd.concat(objs=y, ignore_index=True)
        estimator = Estimator(model_name='xgb')
        estimator.fit(x_train, y_train)

        self.wells = list()
        for well_name, data in self.well_data.items():
            y_adap = estimator.predict_by_test(data['x_train'])
            y_pred = estimator.predict_by_train_test(data['y_train'], data['x_test'])

            # start = data['start_row'][sett.predicate].tolist()[0]
            #
            # fact = list()
            # model = list()
            # y_fact = start
            # y_model = start
            # for i in data['y_train'].index:
            #     y_fact = y_fact * (1 + data['y_train'].loc[i])
            #     y_model = y_fact * (1 + y_adap.loc[i])
            #     fact.append(y_fact)
            #     model.append(y_model)
            # data['y_train'] = Series(fact, data['y_train'].index)
            # y_adap = Series(model, data['y_train'].index)
            #
            # fact = list()
            # model = list()
            # y_fact = data['y_train'].iloc[-1]
            # y_model = y_fact
            # for i in data['y_test'].index:
            #     y_fact = y_fact * (1 + data['y_test'].loc[i])
            #     y_model = y_model * (1 + y_pred.loc[i])
            #     fact.append(y_fact)
            #     model.append(y_model)
            # data['y_test'] = Series(fact, data['y_test'].index)
            # y_pred = Series(model, data['y_test'].index)

            # data['y_train'].plot()
            # y_adap.plot()
            # data['y_test'].plot()
            # y_pred.plot()
            # plt.show()

            # data['x_train']['new'] = y_adap
            # data['x_test']['new'] = y_pred
            well = Well(well_name, data)
            self.wells.append(well)
            print(well_name)

        self.y_dev = self._calc_average_relative_deviations(self.wells)

    @staticmethod
    def _calc_average_relative_deviations(wells: List[Well]) -> Series:
        y_dev = []
        index = []
        well_number = len(wells)
        for i in range(sett.forecast_days_number):
            yd = 0
            for well in wells:
                yd += well.y_dev.iloc[i]
            yd /= well_number
            y_dev.append(yd)
            index.append(i + 1)
        y_dev = Series(y_dev, index, name='Отн. отклонение дебита жидкости от факта, %')
        return y_dev
