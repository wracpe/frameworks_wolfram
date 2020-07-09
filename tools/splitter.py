import settings as sett

from pandas import DataFrame


class Splitter(object):

    _fold_samples_number = sett.forecast_days_number

    def __init__(self, df: DataFrame):
        self.df = df
        self.train_test_pairs = list()
        self.pair_number: int
        self._total_samples_number: int
        self._k_int: int
        self._k_frac: float

        self._calc_fold_number()
        self._create_train_test_pair()
        self._check_last_pair_existing()
        self._calc_pair_number()

    def _calc_fold_number(self):
        self._total_samples_number = len(self.df.index)
        k = self._total_samples_number / self._fold_samples_number
        self._k_int = int(k)
        self._k_frac = k - int(k)

    def _create_train_test_pair(self):
        for i in range(2, self._k_int + 1):
            samples_number = self._fold_samples_number * i
            df_train = self.df.head(samples_number - self._fold_samples_number)
            df_test = self.df.tail(self._fold_samples_number)
            self.train_test_pairs.append({'train': df_train, 'test': df_test})

    def _check_last_pair_existing(self):
        if self._k_frac != 0:
            int_part = self._fold_samples_number * self._k_int
            frac_part = self._total_samples_number - int_part
            df_train = self.df.head(int_part)
            df_test = self.df.tail(frac_part)
            self.train_test_pairs.append({'train': df_train, 'test': df_test})

    def _calc_pair_number(self):
        self.pair_number = len(self.train_test_pairs)
