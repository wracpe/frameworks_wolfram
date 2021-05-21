import datetime
import jsonpickle
import pandas as pd
from typing import Dict, List, Tuple

from config_field import ConfigField
from data_objects.field import Field
from tools.wrapper_estimator import WrapperEstimator
from wrapper_well import WrapperWell


class WrapperField(object):

    def __init__(self, config_field: ConfigField):
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
        self._wrappers_wells = []
        self._create_field_estimator()
        for well_name_ois, data in self._well_data.items():
            print(well_name_ois)
            data = self._add_preview_forecast_as_feature(data)
            well = WrapperWell(self.config_field, well_name_ois, data)
            self._wrappers_wells.append(well)

    def _create_field_estimator(self) -> None:
        self._wrapper_estimator = WrapperEstimator(self.config_field, self.config_field.estimator_name_field)
        self._wrapper_estimator.fit(self._x_train, self._y_train)

    def _add_preview_forecast_as_feature(self, data) -> Dict[str, pd.DataFrame]:
        y_adap = self._wrapper_estimator.predict_by_test(data['x_train'])
        y_pred = self._wrapper_estimator.predict_by_train_test(data['y_train'], data['x_test'])
        data['x_train']['preview_forecast'] = y_adap
        data['x_test']['preview_forecast'] = y_pred
        return data

    @staticmethod
    def _calc_average_relative_deviations(wells: List[WrapperWell]) -> pd.Series:
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

    def __init__(self, config_field: ConfigField, well_name_ois: str):
        self._config_field = config_field
        self._well_name_ois = well_name_ois
        self._data = {}

    def get_data(self) -> Dict[str, pd.DataFrame]:
        self._read_data()
        self._add_features()
        self._split_train_test()
        return self._data

    def _read_data(self) -> None:
        self._df = pd.read_csv(self._config_field.path_data / f'chess_{self._well_name_ois}.csv')
        self._df['dt'] = self._df['dt'].apply(self._convert_day_date)
        self._df.set_index(keys='dt', drop=True, inplace=True, verify_integrity=False)
        self._df.drop(columns=['wc_fact', 'status', 'event', 'work_time'], inplace=True)

    def _add_features(self) -> None:
        for param in [self._config_field.predicate, self._config_field.predictor]:
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

    def _split_train_test(self) -> None:
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
