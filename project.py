import pandas as pd

import settings as sett

from typing import Dict, List

from tools.estimator import Estimator
from well import Well


class Project(object):

    def __init__(self, well_data: Dict[str, Dict]):
        self.wells: List[Well]
        self.y_dev: pd.Series
        self.well_data = well_data
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
            data['x_train']['new'] = y_adap
            data['x_test']['new'] = y_pred
            well = Well(well_name, data)
            self.wells.append(well)
            print(well_name)

        self.y_dev = self._calc_average_relative_deviations(self.wells)

    @staticmethod
    def _calc_average_relative_deviations(wells: List[Well]) -> pd.Series:
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
        y_dev = pd.Series(y_dev, index, name='Отн. отклонение дебита жидкости от факта, %')
        return y_dev
