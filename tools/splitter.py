import pandas as pd


class Splitter(object):

    _fold_samples_number = sett.forecast_days_number
    _r = 2

    def __init__(self, x: pd.DataFrame, y: pd.DataFrame):
        self.x = x
        self.y = y

        self.train_test_pairs = list()
        self.pair_number: int
        self._total_samples_number: int
        self._k_int: int
        self._k_frac: float

        self._calc()
        # self._calc_fold_number()
        # self._create_train_test_pair()
        # self._check_last_pair_existing()
        self._calc_pair_number()

    def _calc(self):
        total_samples_number = len(self.x.index)

        x_train = self.x.head(total_samples_number - self._fold_samples_number)
        y_train = self.y.head(total_samples_number - self._fold_samples_number)

        x_test = self.x.tail(self._fold_samples_number)
        y_test = self.y.tail(self._fold_samples_number)

        self.train_test_pairs.append({'x_train': x_train,
                                      'y_train': y_train,
                                      'x_test': x_test,
                                      'y_test': y_test})

    def _calc_fold_number(self):
        self._total_samples_number = len(self.x.index)
        k = self._total_samples_number / self._fold_samples_number
        self._k_int = int(k)
        self._k_frac = k - int(k)

    def _create_train_test_pair(self):
        for i in range(self._r, self._k_int + 1):
            samples_number = self._fold_samples_number * i
            x_k = self.x.head(samples_number)
            y_k = self.y.head(samples_number)

            x_train = x_k.head(samples_number - self._fold_samples_number)
            y_train = y_k.head(samples_number - self._fold_samples_number)

            x_test = x_k.tail(self._fold_samples_number)
            y_test = y_k.tail(self._fold_samples_number)

            self.train_test_pairs.append({'x_train': x_train,
                                          'y_train': y_train,
                                          'x_test': x_test,
                                          'y_test': y_test})

    def _check_last_pair_existing(self):
        if self._k_frac != 0:
            int_part = self._fold_samples_number * self._k_int
            frac_part = self._total_samples_number - int_part

            x_train = self.x.head(int_part)
            y_train = self.y.head(int_part)

            x_test = self.x.tail(frac_part)
            y_test = self.y.tail(frac_part)

            self.train_test_pairs.append({'x_train': x_train,
                                          'y_train': y_train,
                                          'x_test': x_test,
                                          'y_test': y_test})

    def _calc_pair_number(self):
        self.pair_number = len(self.train_test_pairs)
