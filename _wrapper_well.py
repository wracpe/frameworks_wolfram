import pandas as pd
from sklearn.metrics import mean_absolute_error

from config_field import ConfigField
from _wrapper_estimator import _WrapperEstimator


class _WrapperWell(object):

    def __init__(
            self,
            config_field: ConfigField,
            well_name_ois: str,
            x_train: pd.DataFrame,
            y_train: pd.Series,
            x_test: pd.DataFrame,
            y_test: pd.Series,
    ):
        self._well_name_ois = well_name_ois
        self._config_field = config_field
        self._x_train = x_train
        self._x_test = x_test
        self._y_train_true = y_train
        self._y_test_true = y_test
        self._run()

    def _run(self) -> None:
        self._make_forecast()
        self._calc_deviations()

    def _make_forecast(self) -> None:
        wrapper_estimator = _WrapperEstimator(
            self._config_field,
            self._config_field.estimator_name_well,
        )
        splitter = _Splitter(
            self._x_train,
            self._y_train_true,
            self._config_field.is_deep_grid_search,
            self._config_field.forecast_days_number,
        )
        grid_search = _GridSearch(wrapper_estimator, splitter)
        wrapper_estimator.set_params(**grid_search.params)
        wrapper_estimator.fit(self._x_train, self._y_train_true)
        self._y_train_pred = wrapper_estimator.predict_train(self._x_train)
        self._y_test_pred = wrapper_estimator.predict_test(self._y_train_true, self._x_test)

    def _calc_deviations(self) -> None:
        self.mae_train = mean_absolute_error(self._y_train_true, self._y_train_pred)
        self.mae_test = mean_absolute_error(self._y_test_true, self._y_test_pred)
        # self._calc_relative_deviations()

    def _calc_relative_deviations(self) -> None:
        y_dev = []
        for i in y_true.index:
            y1 = y_true.loc[i]
            y2 = y_pred.loc[i]
            yd = abs(y1 - y2) / max(y1, y2) * 100
            y_dev.append(yd)
        y_dev = pd.DataFrame(y_dev, y_true.index)


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
            print(params, error)
            self._error_params[error] = params

    def _find_params(self) -> None:
        error_min = min(self._error_params.keys())
        self.params = self._error_params[error_min]
        print(self.params)
