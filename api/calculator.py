import pandas as pd
from typing import Dict, List, Tuple, Union

from api.config import Config
from api.well import Well
from _wrapper_estimator import _WrapperEstimator
from _wrapper_well import _WrapperWell


class Calculator(object):

    def __init__(
            self,
            config: Config,
    ):
        self._config = config
        self._run()

    def _run(self) -> None:
        self._predict()

    def _predict(self) -> None:
        self._calculator_rate_liq = _CalculatorRate(
            self._config,
            name_rate_to_predict=Well.NAME_RATE_LIQ,
            name_rate_to_drop=Well.NAME_RATE_OIL,
        )
        self._calculator_rate_oil = _CalculatorRate(
            self._config,
            name_rate_to_predict=Well.NAME_RATE_OIL,
            name_rate_to_drop=Well.NAME_RATE_LIQ,
        )


class _CalculatorRate(object):

    def __init__(
            self,
            config: Config,
            name_rate_to_predict: str,
            name_rate_to_drop: str,
    ):
        self._config = config
        self._name_rate_to_predict = name_rate_to_predict
        self._name_rate_to_drop = name_rate_to_drop
        self._run()

    def _run(self) -> None:
        self._handle_data()
        self._make_forecast_by_wells()

    def _handle_data(self) -> None:
        x, y = [], []
        self._well_data = {}
        for well in self._config.wells:
            handler = _Handler(
                self._config,
                self._name_rate_to_predict,
                self._name_rate_to_drop,
                well.df,
            )
            data = handler.data
            if len(data['y_train']) < self._config.forecast_days_number:
                print(
                    f'Скважина {well.well_name} не будет рассчитана по параметру "{self._name_rate_to_predict}", так'
                    f'как длительность периода адаптации меньше, чем прогноза.'
                )
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
                x_train[Well.NAME_RATE_BASE] = self._group_estimator.predict_train(x_train)
                x_test[Well.NAME_RATE_BASE] = self._group_estimator.predict_test(y_train, x_test)
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

    @property
    def wrapper_wells(self) -> List[_WrapperWell]:
        return self._wrapper_wells


class _Handler(object):

    def __init__(
            self,
            config: Config,
            rate_name_to_predict: str,
            rate_name_to_drop: str,
            df: pd.DataFrame,
    ):
        self._config = config
        self._rate_name_to_predict = rate_name_to_predict
        self._rate_name_to_drop = rate_name_to_drop
        self._df = df
        self._run()

    def _run(self) -> None:
        self._data = {}
        self._drop_excess_rate()
        self._interpolate_gaps()
        self._add_new_features()
        self._split_train_test_x_y()

    def _drop_excess_rate(self) -> None:
        self._df.drop(columns=self._rate_name_to_drop, inplace=True)

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

    def _create_window_features(self, col: str, ws: int) -> None:
        window = self._df[col].rolling(ws)
        col += f'_{ws}'
        self._df[f'{col}_median'] = window.median().shift()
        for q in self._config.quantiles:
            self._df[f'{col}_quantile_{q}'] = window.quantile(q).shift()

    def _split_train_test_x_y(self) -> None:
        df_train = self._df.head(len(self._df.index) - self._config.forecast_days_number)
        df_test = self._df.tail(self._config.forecast_days_number)
        self._data['x_train'], self._data['y_train'] = self._divide_x_y(df_train)
        self._data['x_test'], self._data['y_test'] = self._divide_x_y(df_test)

    def _divide_x_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        x = df.drop(columns=self._rate_name_to_predict)
        y = df[self._rate_name_to_predict]
        return x, y

    @property
    def data(self) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        return self._data
