import pandas as pd
import plotly as pl
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error

from .api.config import Config
from ._wrapper_estimator import _WrapperEstimator


class _WrapperWell(object):

    def __init__(
            self,
            config: Config,
            well_name: int,
            x_train: pd.DataFrame,
            x_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series,
    ):
        self.config = config
        self.well_name = well_name
        self.x_train = x_train
        self.x_test = x_test
        self.y_train_true = y_train
        self.y_test_true = y_test
        self._run()

    def _run(self) -> None:
        self._make_forecast()
        self._calc_deviations()
        _Plotter(
            self.config,
            self.well_name,
            self.x_train,
            self.x_test,
            self.y_train_true,
            self.y_train_pred,
            self.y_test_true,
            self.y_test_pred,
            self.y_dev,
        )

    def _make_forecast(self) -> None:
        wrapper_estimator = _WrapperEstimator(
            self.config,
            self.config.estimator_name_well,
        )
        splitter = _Splitter(
            self.x_train,
            self.y_train_true,
            self.config.is_deep_grid_search,
            self.config.forecast_days_number,
        )
        grid_search = _GridSearch(wrapper_estimator, splitter)
        wrapper_estimator.set_params(grid_search.params)
        wrapper_estimator.fit(self.x_train, self.y_train_true)
        self.y_train_pred = wrapper_estimator.predict_train(self.x_train)
        self.y_test_pred = wrapper_estimator.predict_test(self.y_train_true, self.x_test)

    def _calc_deviations(self) -> None:
        self.mae_train = mean_absolute_error(self.y_train_true, self.y_train_pred)
        self.mae_test = mean_absolute_error(self.y_test_true, self.y_test_pred)
        self._calc_relative_deviations()

    def _calc_relative_deviations(self) -> None:
        y_dev = []
        for i in self.y_test_true.index:
            y1 = self.y_test_true.loc[i]
            y2 = self.y_test_pred.loc[i]
            yd = abs(y1 - y2) / y1
            y_dev.append(yd)
        self.y_dev = pd.Series(y_dev, self.y_test_true.index)


class _Splitter(object):

    def __init__(
            self,
            x_train: pd.DataFrame,
            y_train: pd.Series,
            is_deep_split: bool,
            fold_samples_number: int,
    ):
        self._x = x_train
        self._y = y_train
        self._is_deep_split = is_deep_split
        self._fold_samples_number = fold_samples_number
        self._total_samples_number = len(self._y.index)
        self._run()

    def _run(self) -> None:
        self.train_test_pairs = []
        if self._is_deep_split:
            self._set_fold_number()
            self._create_train_test_pair()
        else:
            self._add_pair(self._total_samples_number, self._fold_samples_number)
        self.pair_number = len(self.train_test_pairs)

    def _set_fold_number(self) -> None:
        fn = self._total_samples_number / self._fold_samples_number
        self._fn_int = int(fn)
        self._fn_frac = fn - self._fn_int

    def _create_train_test_pair(self) -> None:
        for i in range(3, self._fn_int + 1):
            pair_samples_number = self._fold_samples_number * i
            test_samples_number = pair_samples_number - self._fold_samples_number
            self._add_pair(pair_samples_number, test_samples_number)
        self._check_last_pair_existing()

    def _check_last_pair_existing(self) -> None:
        if self._fn_frac != 0:
            pair_samples_number = self._total_samples_number
            test_samples_number = self._total_samples_number - self._fold_samples_number * self._fn_int
            self._add_pair(pair_samples_number, test_samples_number)

    def _add_pair(self, pair_samples_number: int, test_samples_number: int) -> None:
        train_samples_number = pair_samples_number - test_samples_number
        x_pair = self._x.head(pair_samples_number)
        y_pair = self._y.head(pair_samples_number)
        x_train = x_pair.head(train_samples_number)
        y_train = y_pair.head(train_samples_number)
        x_test = x_pair.tail(test_samples_number)
        y_test = y_pair.tail(test_samples_number)
        self.train_test_pairs.append({
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
        })


class _GridSearch(object):

    def __init__(
            self,
            wrapper_estimator: _WrapperEstimator,
            splitter: _Splitter,
    ):
        self._wrapper_estimator = wrapper_estimator
        self._splitter = splitter
        self._run()

    def _run(self) -> None:
        self._calc_grid()
        self._find_params()

    def _calc_grid(self) -> None:
        self._error_params = {}
        for params in self._wrapper_estimator.get_param_grid():
            error = 0
            for pair in self._splitter.train_test_pairs:
                self._wrapper_estimator.set_params(params)
                self._wrapper_estimator.fit(pair['x_train'], pair['y_train'])
                y_pred = self._wrapper_estimator.predict_test(pair['y_train'], pair['x_test'])
                error += mean_absolute_error(pair['y_test'], y_pred)
            error /= self._splitter.pair_number
            self._error_params[error] = params

    def _find_params(self) -> None:
        error_min = min(self._error_params.keys())
        self.params = self._error_params[error_min]
        print(self.params)


class _Plotter(object):

    def __init__(
            self,
            config: Config,
            well_name: int,
            x_train: pd.DataFrame,
            x_test: pd.DataFrame,
            y_train_true: pd.Series,
            y_train_pred: pd.Series,
            y_test_true: pd.Series,
            y_test_pred: pd.Series,
            y_dev: pd.Series,
    ):
        self._config = config
        self._well_name = well_name
        self._x_train = x_train
        self._x_test = x_test
        self._y_train_true = y_train_true
        self._y_train_pred = y_train_pred
        self._y_test_true = y_test_true
        self._y_test_pred = y_test_pred
        self._y_dev = y_dev
        self._run()

    def _run(self) -> None:
        self._prepare()
        self._create_plot()

    def _prepare(self) -> None:
        self._x = pd.concat(objs=[self._x_train, self._x_test], axis=0)
        self._x_ax = self._x.index
        self._x_ax_test = self._x_test.index
        self._x_date_test = self._x_train.index[-1]
        self._y_true = pd.concat(objs=[self._y_train_true, self._y_test_true], axis=0)
        self._y_pred = pd.concat(objs=[self._y_train_pred, self._y_test_pred], axis=0)
        self._set_main_predictors()

    def _set_main_predictors(self) -> None:
        cols = self._x.columns
        cols_drop = []
        specific_identifier = '_'
        for col in cols:
            if specific_identifier in col:
                continue
            cols_drop.append(col)
        self._df_predictors = self._x[cols_drop]

    def _create_plot(self) -> None:
        figure = go.Figure(layout=go.Layout(
            font=dict(size=10),
            hovermode='x',
            template='seaborn',
        ))
        fig = make_subplots(
            rows=2,
            cols=2,
            shared_xaxes=True,
            column_width=[0.7, 0.3],
            row_heights=[0.7, 0.3],
            vertical_spacing=0.02,
            horizontal_spacing=0.02,
            figure=figure,
        )
        m = 'markers'
        ml = 'markers+lines'
        mark = dict(size=3)
        line = dict(width=1)

        trace = go.Scatter(name='true', x=self._x_ax, y=self._y_true, mode=m, marker=mark)
        fig.add_trace(trace, row=1, col=1)

        trace = go.Scatter(name='pred', x=self._x_ax, y=self._y_pred, mode=ml, marker=mark, line=line)
        fig.add_trace(trace, row=1, col=1)

        for col in self._df_predictors.columns:
            trace = go.Scatter(name=col, x=self._x_ax, y=self._df_predictors[col], mode=m, marker=mark)
            fig.add_trace(trace, row=2, col=1)

        trace = go.Scatter(name='true', x=self._x_ax_test, y=self._y_test_true, mode=m, marker=mark)
        fig.add_trace(trace, row=1, col=2)

        trace = go.Scatter(name='pred', x=self._x_ax_test, y=self._y_test_pred, mode=ml, marker=mark, line=line)
        fig.add_trace(trace, row=1, col=2)

        trace = go.Scatter(name='dev', x=self._x_ax_test, y=self._y_dev, mode=ml, marker=mark, line=line)
        fig.add_trace(trace, row=2, col=2)

        fig.add_vline(row=1, col=1, x=self._x_date_test, line_width=2, line_dash='dash')
        fig.add_vline(row=2, col=1, x=self._x_date_test, line_width=2, line_dash='dash')

        path_str = str(self._config.path_save)
        well_name = self._well_name
        predicate = self._config.predicate
        file = f'{path_str}\\{well_name}_{predicate}.png'
        pl.io.write_image(fig, file=file, width=1450, height=700, scale=2, engine='kaleido')
