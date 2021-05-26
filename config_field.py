import pathlib
from typing import List


class ConfigField(object):

    predictor = 'p'
    _path = pathlib.Path.cwd() / 'data'

    def __init__(
            self,
            name: str,
            predicate: str,
            estimator_name_field: str,
            estimator_name_well: str,
            is_deep_grid_search: bool,
            window_sizes: List[int],
            quantiles: List[float],
            well_names_ois: List[str] = None,
    ):
        self.name = name
        self.predicate = predicate
        self.estimator_name_field = estimator_name_field
        self.estimator_name_well = estimator_name_well
        self.is_deep_grid_search = is_deep_grid_search
        self.window_sizes = window_sizes
        self.quantiles = quantiles
        self.well_names_ois = well_names_ois
        self._set_paths()

    def _set_paths(self):
        field_path = self._path / self.name
        self.path_data = field_path / 'packed_data'
        self.path_results = field_path / 'results'
        self.path_json_dump = self.path_data / f'{self.name}_dump.json'
