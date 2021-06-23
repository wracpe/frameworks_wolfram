import datetime
import pandas as pd
import plotly as pl
import plotly.graph_objs as go
from typing import Dict, Tuple, Union

from .api.config import Config
from ._wrapper_estimator import _WrapperEstimator
from ._wrapper_well import _WrapperWell


class _Calculator(object):

    def __init__(
            self,
            config: Config,
    ):
        self._config = config
        self._run()

    def _run(self) -> None:
        self._check_create_dir_existence()
        self._handle_data()
        self._make_forecast_by_wells()
        self._save_well_results()
        self._calc_deviations()
        _Plotter(
            self._config,
            self._y_dev,
        )

    def _check_create_dir_existence(self) -> None:
        run_name = f'results_{str(datetime.date.today())}_wolfram_{str(self._config.wells[0].df.index[-self._config.forecast_days_number])}'
        path_save = self._config.path_save / run_name
        if not path_save.exists():
            path_save.mkdir(parents=False)
        self._config.path_save = path_save

    def _handle_data(self) -> None:
        x, y = [], []
        self._well_data = {}
        for well in self._config.wells:
            handler = _HandlerByWellData(
                self._config,
                well.df,
            )
            data = handler.data
            if len(data['y_train']) < self._config.forecast_days_number:
                continue
            x.append(data['x_train'])
            y.append(data['y_train'])
            self._well_data[well.well_name] = data
        self._x_train = pd.concat(objs=x, ignore_index=True)
        self._y_train = pd.concat(objs=y, ignore_index=True)

    def _make_forecast_by_wells(self) -> None:
        self._wrapper_wells = []
        self._create_group_estimator()
        for well_name, data in self._well_data.items():
            print(well_name)
            try:
                x_train, y_train, x_test, y_test = data.values()
                col = 'Predicate by group estimator '
                x_train[col] = self._group_estimator.predict_train(x_train)
                x_test[col] = self._group_estimator.predict_test(y_train, x_test)
                wrapper_well = _WrapperWell(
                    self._config,
                    well_name,
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                )
                self._wrapper_wells.append(wrapper_well)
            except Exception as exc:
                print(exc)
                continue

    def _create_group_estimator(self) -> None:
        self._group_estimator = _WrapperEstimator(
            self._config,
            self._config.estimator_name_group,
        )
        self._group_estimator.fit(self._x_train, self._y_train)

    def _save_well_results(self) -> None:
        dfs = []
        for wrapper_well in self._wrapper_wells:
            name = wrapper_well.well_name
            df = pd.DataFrame(index=wrapper_well.y_test_true.index)
            df[f'{name}_true'] = wrapper_well.y_test_true
            df[f'{name}_pred'] = wrapper_well.y_test_pred
            dfs.append(df)
        df = pd.concat(objs=dfs, axis=1)
        df.to_excel(self._config.path_save / f'well_results_{self._config.predicate}.xlsx')

    def _calc_deviations(self) -> None:
        self._y_dev = pd.Series(dtype='float64')
        well_number = len(self._wrapper_wells)
        for day in range(self._config.forecast_days_number):
            yd = 0
            for wrapper_well in self._wrapper_wells:
                yd += wrapper_well.y_dev.iloc[day]
            yd /= well_number
            self._y_dev.loc[day] = yd


class _HandlerByWellData(object):

    def __init__(
            self,
            config: Config,
            df: pd.DataFrame,
    ):
        self._config = config
        self._df = df
        self._run()

    @property
    def data(self) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        return self._data

    def _run(self) -> None:
        self._data = {}
        self._interpolate_gaps()
        self._add_new_features()
        self._split_train_test_x_y()

    def _interpolate_gaps(self) -> None:
        index = self._df.index
        self._df.reset_index(drop=True, inplace=True)
        self._df.interpolate(method='ffill', inplace=True)
        self._df.interpolate(method='bfill', inplace=True)
        self._df.index = index

    def _add_new_features(self) -> None:
        for col in self._df.columns:
            for ws in self._config.window_sizes:
                self._create_window_features(col, ws)
        self._df.dropna(inplace=True)

    def _create_window_features(self, col: str, window_size: int) -> None:
        window = self._df[col].rolling(window_size)
        col += f'_{window_size}'
        self._df[f'{col}_median'] = window.median().shift()
        for q in self._config.quantiles:
            self._df[f'{col}_quantile_{q}'] = window.quantile(q).shift()

    def _split_train_test_x_y(self) -> None:
        df_train = self._df.head(len(self._df.index) - self._config.forecast_days_number)
        df_test = self._df.tail(self._config.forecast_days_number)
        self._data['x_train'], self._data['y_train'] = self._divide_x_y(df_train)
        self._data['x_test'], self._data['y_test'] = self._divide_x_y(df_test)

    def _divide_x_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        x = df.drop(columns=self._config.predicate)
        y = df[self._config.predicate]
        return x, y

    @staticmethod
    def _convert_day_date(x: str) -> datetime.date:
        return datetime.datetime.strptime(x, '%Y-%m-%d').date()


class _Plotter(object):

    def __init__(
            self,
            config: Config,
            y_dev: pd.Series,
    ):
        self._config = config
        self._y_dev = y_dev
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

        trace = go.Scatter(name='true', x=self._y_dev.index, y=self._y_dev, mode=ml, marker=mark, line=line)
        figure.add_trace(trace)

        path_str = str(self._config.path_save)
        file = f'{path_str}\\performance_{self._config.predicate}.png'
        pl.io.write_image(figure, file=file, width=1450, height=700, scale=2, engine='kaleido')
