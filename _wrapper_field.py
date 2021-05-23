import datetime
import jsonpickle
import pandas as pd
from typing import Dict, List, Tuple

from config_field import ConfigField
from data_objects.field import Field
from _wrapper_estimator import _WrapperEstimator
from _wrapper_well import _WrapperWell


class _WrapperField(object):

    def __init__(
            self,
            config_field: ConfigField,
    ):
        self.config_field = config_field
        self._run()

    def _run(self) -> None:
        self._create_field_from_json_dump()
        self._read_and_prepare_data()
        self._make_forecast_by_wells()
        self._calc_average_relative_deviations()

    def _create_field_from_json_dump(self) -> None:
        with open(self.config_field.path_json_dump, 'r') as f:
            json_dump = f.read()
        self._field = jsonpickle.decode(json_dump, classes=Field)

    def _read_and_prepare_data(self) -> None:
        x = []
        y = []
        self._well_data = {}
        for well in self._field.wells:
            data_handler_well = _DataHandlerWell(self.config_field, well.name_ois)
            data = data_handler_well.get_data()
            x.append(data['x_train'])
            y.append(data['y_train'])
            self._well_data[well.name_ois] = data
        self._x_train = pd.concat(objs=x, ignore_index=True)
        self._y_train = pd.concat(objs=y, ignore_index=True)

    def _make_forecast_by_wells(self) -> None:
        self._wrapper_wells = []
        self._create_field_estimator()
        for well_name_ois, data in self._well_data.items():
            print(well_name_ois)
            data = self._add_field_forecast(data)
            wrapper_well = _WrapperWell(
                self.config_field,
                well_name_ois,
                data,
            )
            self._wrapper_wells.append(wrapper_well)

    def _create_field_estimator(self) -> None:
        self._wrapper_estimator = _WrapperEstimator(
            self.config_field,
            self.config_field.estimator_name_field,
        )
        self._wrapper_estimator.fit(self._x_train, self._y_train)

    def _add_field_forecast(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        data['x_train']['ff'] = self._wrapper_estimator.predict_train(data['x_train'])
        data['x_test']['ff'] = self._wrapper_estimator.predict_test(data['y_train'], data['x_test'])
        return data

    @staticmethod
    def _calc_average_relative_deviations(wells: List[_WrapperWell]) -> pd.Series:
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


class _DataHandlerWell(object):

    def __init__(
            self,
            config_field: ConfigField,
            well_name_ois: str,
    ):
        self._config_field = config_field
        self._well_name_ois = well_name_ois
        self._data = {}

    def get_data(self) -> Dict[str, pd.DataFrame]:
        self._read_data()
        self._add_features()
        self._split_train_test_x_y()
        return self._data

    def _read_data(self) -> None:
        self._param_names = [
            self._config_field.predictor,
            self._config_field.predicate,
        ]
        self._df = pd.read_csv(
            filepath_or_buffer=self._config_field.path_data / f'chess_{self._well_name_ois}.csv',
            index_col=True,
            usecols=self._param_names,
        )
        self._df.index = self._df.index.map(self._convert_day_date)

    def _add_features(self) -> None:
        for param in self._param_names:
            for ws in self._config_field.window_sizes:
                self._create_window_features(param, ws)
        self._df.dropna(inplace=True)

    def _create_window_features(self, param: str, window_size: int) -> None:
        window = self._df[param].rolling(window_size)
        param += f'_{window_size}'
        self._df[f'{param}_mean'] = window.mean().shift()
        self._df[f'{param}_var'] = window.var().shift()
        for q in self._config_field.quantiles:
            self._df[f'{param}_quantile_{q}'] = window.quantile(q).shift()

    def _split_train_test_x_y(self) -> None:
        df_train = self._df.head(len(self._df.index) - self._config_field.forecast_days_number)
        df_test = self._df.tail(self._config_field.forecast_days_number)
        self._data['x_train'], self._data['y_train'] = self._divide_x_y(df_train)
        self._data['x_test'], self._data['y_test'] = self._divide_x_y(df_test)

    def _divide_x_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        x = df.drop(columns=self._config_field.predicate)
        y = df[self._config_field.predicate]
        return x, y

    @staticmethod
    def _convert_day_date(x: str) -> datetime.date:
        return datetime.datetime.strptime(x, '%Y-%m-%d').date()
