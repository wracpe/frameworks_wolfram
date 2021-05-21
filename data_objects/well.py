import datetime
from typing import Any, Dict, List

from data_objects.bounds import Bounds


class Well(object):

    def __init__(
            self,
            name_ois: int,
            name_geo: str,
            kind_name: str,
            kind_code: int,
            formation_names: List[str],
            h: float,
            phi: float,
            ct: float,
            mul: float,
            bl: float,
            rhoo: float,
            date_start: datetime.date,
            date_test: datetime.date,
            date_end: datetime.date,
            cuml_start: float,
            cuml_test: float,
            cumo_test: float,
            p_name: str,
            bounds: Dict[datetime.date, Bounds],
    ):
        self.name_ois = name_ois
        self.name_geo = name_geo
        self.kind_name = kind_name
        self.kind_code = kind_code
        self.formation_names = formation_names
        self.h = h
        self.phi = phi
        self.ct = ct
        self.mul = mul
        self.bl = bl
        self.rhoo = rhoo
        self.date_start = date_start
        self.date_test = date_test
        self.date_end = date_end
        self.cuml_start = cuml_start
        self.cuml_test = cuml_test
        self.cumo_test = cumo_test
        self.p_name = p_name
        self.bounds = bounds

    @staticmethod
    def _convert_str_to_date(date_str: str) -> datetime.date:
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()

    def __getstate__(self) -> Dict[str, Any]:
        state = {
            'name_ois': str(self.name_ois),
            'name_geo': self.name_geo,
            'kind_name': self.kind_name,
            'kind_code': self.kind_code,
            'formation_names': self.formation_names,
            'h': self.h,
            'phi': self.phi,
            'ct': self.ct,
            'mul': self.mul,
            'bl': self.bl,
            'rhoo': self.rhoo,
            'date_start': str(self.date_start),
            'date_test': str(self.date_test),
            'date_end': str(self.date_end),
            'cuml_start': float(self.cuml_start),
            'cuml_test': float(self.cuml_test),
            'cumo_test': float(self.cumo_test),
            'p_name': str(self.p_name),
            'bounds': {str(key): value for key, value in self.bounds.items()},
        }
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.name_ois = state['name_ois']
        self.name_geo = state['name_geo']
        self.kind_name = state['kind_name']
        self.kind_code = state['kind_code']
        self.formation_names = state['formation_names']
        self.h = state['h']
        self.phi = state['phi']
        self.ct = state['ct']
        self.mul = state['mul']
        self.bl = state['bl']
        self.rhoo = state['rhoo']
        self.date_start = self._convert_str_to_date(state['date_start'])
        self.date_test = self._convert_str_to_date(state['date_test'])
        self.date_end = self._convert_str_to_date(state['date_end'])
        self.cuml_start = state['cuml_start']
        self.cuml_test = state['cuml_test']
        self.cumo_test = state['cumo_test']
        self.p_name = state['p_name']
        self.bounds = {self._convert_str_to_date(key): value for key, value in state['bounds'].items()}
