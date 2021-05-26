import datetime
import jsonpickle
import pandas as pd
import plotly as pl
import plotly.graph_objs as go
from typing import Dict, Tuple, Union

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
        self._correct_well_data()
        self._make_forecast_by_wells()
        self._calc_deviations()
        self._save_well_results()
        _Plotter(self)

    def _create_field_from_json_dump(self) -> None:
        with open(self.config_field.path_json_dump, 'r') as f:
            json_dump = f.read()
        self.field = jsonpickle.decode(json_dump, classes=Field)
        self.dates_test = pd.date_range(self.field.date_test, self.field.date_end).date
        self.forecast_days_number = len(self.dates_test)

    def _read_and_prepare_data(self) -> None:
        x, y = [], []
        self.well_data = {}
        for well in self.field.wells:
            data_handler_well = _DataHandlerWell(self.config_field, self.forecast_days_number, well.name_ois)
            data = data_handler_well.get_data()
            if len(data['y_train']) < self.forecast_days_number:
                continue
            x.append(data['x_train'])
            y.append(data['y_train'])
            self.well_data[well.name_ois] = data
        self.x_train = pd.concat(objs=x, ignore_index=True)
        self.y_train = pd.concat(objs=y, ignore_index=True)

    def _correct_well_data(self) -> None:
        if self.config_field.well_names_ois is not None:
            self.well_data = {name: self.well_data[name] for name in self.config_field.well_names_ois}

    def _make_forecast_by_wells(self) -> None:
        self.wrapper_wells = []
        self._create_field_estimator()
        for well_name_ois, data in self.well_data.items():
            print(well_name_ois)
            x_train, y_train, x_test, y_test = data.values()
            x_train['q_by_field'] = self.field_estimator.predict_train(x_train)
            x_test['q_by_field'] = self.field_estimator.predict_test(y_train, x_test)
            wrapper_well = _WrapperWell(
                self.config_field,
                self.forecast_days_number,
                well_name_ois,
                x_train,
                y_train,
                x_test,
                y_test,
            )
            self.wrapper_wells.append(wrapper_well)

    def _create_field_estimator(self) -> None:
        self.field_estimator = _WrapperEstimator(
            self.config_field,
            self.config_field.estimator_name_field,
        )
        self.field_estimator.fit(self.x_train, self.y_train)

    def _calc_deviations(self) -> None:
        self.y_dev = pd.Series(index=self.dates_test)
        well_number = len(self.wrapper_wells)
        for date in self.dates_test:
            yd = 0
            for wrapper_well in self.wrapper_wells:
                yd += wrapper_well.y_dev.loc[date]
            yd /= well_number
            self.y_dev.loc[date] = yd

    def _save_well_results(self) -> None:
        df = pd.DataFrame(index=self.dates_test)
        for wrapper_well in self.wrapper_wells:
            name = wrapper_well.well_name_ois
            df[f'{name}_true'] = wrapper_well.y_test_true
            df[f'{name}_pred'] = wrapper_well.y_test_pred
        df.to_excel(self.config_field.path_results / f'well_results_{self.config_field.predicate}.xlsx')


class _DataHandlerWell(object):

    def __init__(
            self,
            config_field: ConfigField,
            forecast_days_number: int,
            well_name_ois: str,
    ):
        self._config_field = config_field
        self._forecast_days_number = forecast_days_number
        self._well_name_ois = well_name_ois
        self._data = {}

    def get_data(self) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
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
            usecols=['dt'] + self._param_names,
        )
        self._df.interpolate(method='nearest', inplace=True)
        self._df.set_index(keys='dt', inplace=True, verify_integrity=True)
        self._df.index = self._df.index.map(self._convert_day_date)

    def _add_features(self) -> None:
        for param in self._param_names:
            for ws in self._config_field.window_sizes:
                self._create_window_features(param, ws)
        self._df.dropna(inplace=True)

    def _create_window_features(self, param: str, window_size: int) -> None:
        window = self._df[param].rolling(window_size)
        param += f'_{window_size}'
        self._df[f'{param}_median'] = window.median().shift()
        for q in self._config_field.quantiles:
            self._df[f'{param}_quantile_{q}'] = window.quantile(q).shift()

    def _split_train_test_x_y(self) -> None:
        df_train = self._df.head(len(self._df.index) - self._forecast_days_number)
        df_test = self._df.tail(self._forecast_days_number)
        self._data['x_train'], self._data['y_train'] = self._divide_x_y(df_train)
        self._data['x_test'], self._data['y_test'] = self._divide_x_y(df_test)

    def _divide_x_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        x = df.drop(columns=self._config_field.predicate)
        y = df[self._config_field.predicate]
        y = pd.Series(y.array, y.index)
        return x, y

    @staticmethod
    def _convert_day_date(x: str) -> datetime.date:
        return datetime.datetime.strptime(x, '%Y-%m-%d').date()


class _Plotter(object):

    def __init__(
            self,
            wrapper_field: _WrapperField,
    ):
        self._wrapper_field = wrapper_field
        self._run()

    def _run(self) -> None:
        figure = go.Figure(layout=go.Layout(
            font=dict(size=10),
            hovermode='x',
            template='seaborn',
        ))
        ml = 'markers+lines'
        mark = dict(size=3)
        line = dict(width=1)

        y_dev = self._wrapper_field.y_dev
        trace = go.Scatter(name='true', x=y_dev.index, y=y_dev, mode=ml, marker=mark, line=line)
        figure.add_trace(trace)

        path_str = str(self._wrapper_field.config_field.path_results)
        file = f'{path_str}\\performance.png'
        pl.io.write_image(figure, file=file, width=1450, height=700, scale=2, engine='kaleido')
