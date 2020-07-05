import settings as sett

from pandas import Series
from typing import List

from well import Well


class MathTools(object):

    @staticmethod
    def calc_relative_deviations(y_true: Series, y_pred: Series) -> Series:
        y_dev = []
        for i in y_true.index:
            y1 = y_true.loc[i]
            y2 = y_pred.loc[i]
            yd = abs(y1 - y2) / max(y1, y2)
            y_dev.append(yd)
        y_dev = Series(y_dev, y_true.index)
        return y_dev

    @staticmethod
    def calc_average_relative_deviations(wells: List[Well]) -> Series:
        y_dev = []
        well_number = len(wells)
        for i in range(sett.forecast_days_number):
            yd = 0
            for well in wells:
                yd += well.y_dev.iloc[i]
            yd /= well_number
            y_dev.append(yd)
        y_dev = Series(y_dev, range(sett.forecast_days_number))
        return y_dev
