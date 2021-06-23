import pathlib
from typing import List

from .well import Well


class Config(object):

    def __init__(
            self,
            path_save: pathlib.Path,
            wells: List[Well],
            predicate: str,
            forecast_days_number: int,
            estimator_name_group: str = 'svr',
            estimator_name_well: str = 'ela',
            is_deep_grid_search: bool = False,
            window_sizes: List[int] = None,
            quantiles: List[float] = None,
    ):
        if window_sizes is None:
            window_sizes = [3, 7, 15, 30]
        if quantiles is None:
            quantiles = [0.1, 0.3]

        self.path_save = path_save
        self.wells = wells
        self.predicate = predicate
        self.forecast_days_number = forecast_days_number
        self.estimator_name_group = estimator_name_group
        self.estimator_name_well = estimator_name_well
        self.is_deep_grid_search = is_deep_grid_search
        self.window_sizes = window_sizes
        self.quantiles = quantiles
