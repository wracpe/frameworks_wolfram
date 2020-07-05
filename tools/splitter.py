import settings as sett

from pandas import DataFrame


class Splitter(object):

    _df_train: DataFrame
    _df_test: DataFrame

    @classmethod
    def split_train_test(cls, df: DataFrame) -> (DataFrame, DataFrame):
        days_number = len(df.index)
        cls._df_train = df.head(days_number - sett.forecast_days_number)
        cls._df_test = df.tail(sett.forecast_days_number)
        cls._clear_test()
        return cls._df_train, cls._df_test

    @classmethod
    def _clear_test(cls):
        pass
