import datetime
import pathlib
from typing import Any, Dict, List

from data_objects.well import Well


class Field(object):

    _path = pathlib.Path.cwd() / 'data'

    def __init__(
            self,
            name: str,
            date_start: datetime.date,
            date_test: datetime.date,
            date_end: datetime.date,
            shops: List[str],
            well_names_ois: List[int] = None,
            imputer_estimator: str = 'knn',
    ):
        self.path_data = self._path / name
        self.name = name
        self.date_start = date_start
        self.date_test = date_test
        self.date_end = date_end
        self.shops = shops
        self.well_names_ois = well_names_ois
        self.imputer_estimator = imputer_estimator
        self._wells = None

    @property
    def wells(self) -> List[Well]:
        return self._wells

    @wells.setter
    def wells(self, wells: List[Well]) -> None:
        self._wells = wells

    @staticmethod
    def _convert_str_to_date(date_str: str) -> datetime.date:
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()

    def __getstate__(self) -> Dict[str, Any]:
        state = {
            'path_data': str(self.path_data),
            'name': self.name,
            'date_start': str(self.date_start),
            'date_test': str(self.date_test),
            'date_end': str(self.date_end),
            'shops': self.shops,
            'well_names_ois': self.well_names_ois,
            'imputer_estimator': self.imputer_estimator,
            'wells': self._wells,
        }
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.path_data = pathlib.Path(state['path_data'])
        self.name = state['name']
        self.date_start = self._convert_str_to_date(state['date_start'])
        self.date_test = self._convert_str_to_date(state['date_test'])
        self.date_end = self._convert_str_to_date(state['date_end'])
        self.shops = state['shops']
        self.well_names_ois = state['well_names_ois']
        self.imputer_estimator = state['imputer_estimator']
        self.wells = state['wells']
