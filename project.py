from pandas import DataFrame, Series
from typing import Dict, List

from tools.math_tools import MathTools
from tools.plotter import Plotter
from well import Well


class Project(object):

    def __init__(self, dfs: Dict[str, DataFrame]):
        self.dfs = dfs
        self.wells: List[Well]
        self.y_dev: Series

        self._calc_wells()
        Plotter.create_project_plot(self)

    def _calc_wells(self):
        self.wells = []
        for well_name, df in self.dfs.items():
            well = Well(well_name, df)
            self.wells.append(well)
        self.y_dev = MathTools.calc_average_relative_deviations(self.wells)
