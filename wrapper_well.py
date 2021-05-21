import pandas as pd
from sklearn.metrics import mean_absolute_error
from typing import Dict

from config_field import ConfigField
from tools.wrapper_estimator import WrapperEstimator


class WrapperWell(object):

    def __init__(
            self,
            config_field: ConfigField,
            well_name_ois: str,
            data: Dict[str, pd.DataFrame],
    ):
        self._config_field = config_field
        self._well_name_ois = well_name_ois
        self._data = data
        self._run()

    def _run(self) -> None:
        self._make_forecast()
        self._calc_deviations()

    def _make_forecast(self) -> None:
        wrapper_estimator = WrapperEstimator(self._config_field, self._config_field.estimator_name_well)
        splitter = _Splitter(
            self._data['x_train'],
            self._data['y_train'],
            self._config_field.is_deep_grid_search,
            self._config_field.forecast_days_number,
        )
        grid_search = _GridSearch(wrapper_estimator, splitter)
        wrapper_estimator.estimator.model.set_params(**grid_search.params)
        wrapper_estimator.fit(self._data['x_train'], self._data['y_train'])
        self.y_adap = wrapper_estimator.predict_by_test(self._data['x_train'])
        self.y_pred = wrapper_estimator.predict_by_train_test(self._data['y_train'], self._data['x_test'])

    def _calc_deviations(self):
        # start = self.data['start_row'][sett.predicate].tolist()[0]

        # fact = list()
        # model = list()
        # y_fact = start
        # y_model = start
        # for i in self.data['y_train'].index:
        #     y_fact = y_fact * (1 + self.data['y_train'].loc[i])
        #     y_model = y_fact * (1 + self.y_adap.loc[i])
        #     fact.append(y_fact)
        #     model.append(y_model)
        # self.data['y_train'] = pd.Series(fact, self.data['y_train'].index)
        # self.y_adap = pd.Series(model, self.data['y_train'].index)

        # fact = list()
        # model = list()
        # y_fact = self.data['y_train'].iloc[-1]
        # y_model = y_fact
        # for i in self.data['y_test'].index:
        #     y_fact = y_fact * (1 + self.data['y_test'].loc[i])
        #     y_model = y_model * (1 + self.y_pred.loc[i])
        #     fact.append(y_fact)
        #     model.append(y_model)
        # self.data['y_test'] = pd.Series(fact, self.data['y_test'].index)
        # self.y_pred = pd.Series(model, self.data['y_test'].index)

        self.y_dev = self._calc_relative_deviations(y_true=self._data['y_test'], y_pred=self.y_pred)
        self.MAE_adap = mean_absolute_error(y_true=self._data['y_train'], y_pred=self.y_adap)
        self.MAE_fore = mean_absolute_error(y_true=self._data['y_test'], y_pred=self.y_pred)

    @staticmethod
    def _calc_relative_deviations(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
        y_dev = []
        for i in y_true.index:
            y1 = y_true.loc[i]
            y2 = y_pred.loc[i]
            yd = abs(y1 - y2) / max(y1, y2) * 100
            y_dev.append(yd)
        y_dev = pd.Series(y_dev, y_true.index)
        return y_dev


class _Splitter(object):

    def __init__(
            self,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
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

    def __init__(self, wrapper_estimator: WrapperEstimator, splitter: _Splitter):
        self._wrapper_estimator = wrapper_estimator
        self._splitter = splitter
        self._run()

    def _run(self) -> None:
        self._calc_grid()
        self._find_params()

    def _calc_grid(self) -> None:
        self._error_params = {}
        for params in self._wrapper_estimator.estimator.param_grid:
            error = 0
            for pair in self._splitter.train_test_pairs:
                self._wrapper_estimator.estimator.model.set_params(**params)
                self._wrapper_estimator.fit(pair['x_train'], pair['y_train'])
                y_pred = self._wrapper_estimator.predict_by_train_test(pair['y_train'], pair['x_test'])
                error += mean_absolute_error(pair['y_test'], y_pred)
            error /= self._splitter.pair_number
            print(params, error)
            self._error_params[error] = params

    def _find_params(self) -> None:
        error_min = min(self._error_params.keys())
        self.params = self._error_params[error_min]
        print(self.params)
