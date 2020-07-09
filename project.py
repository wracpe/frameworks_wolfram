import settings as sett

from pandas import DataFrame, Series
from typing import Dict, List

from well import Well


class Project(object):

    def __init__(self, dfs: Dict[str, DataFrame]):
        self.dfs = dfs

        self.wells: List[Well]
        self.y_dev: Series

        self._calc_wells()

    def _calc_wells(self):
        self.wells = []
        for well_name, df in self.dfs.items():
            print(well_name)
            well = Well(well_name, df)
            self.wells.append(well)
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
