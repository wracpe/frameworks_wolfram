import settings as sett

from pandas import DataFrame


class WindowFeatureGenerator(object):

    _df: DataFrame
    _column: str
    _window_sizes = sett.window_sizes
    _quantiles = sett.quantiles

    @classmethod
    def run(cls, df: DataFrame) -> DataFrame:
        cls._df = df.copy()
        for column in [sett.predictor, sett.predicate]:
        # for column in [sett.predicate]:
            cls._add_features(column)
        return cls._df

    @classmethod
    def _add_features(cls, column: str):
        for ws in cls._window_sizes:
            cls._column = column
            cls._create_window_features(ws)
        cls._df.dropna(inplace=True)

    @classmethod
    def _create_window_features(cls, window_size: float):
        window = cls._df[cls._column].rolling(window_size)
        cls._column += f'_{window_size}'
        cls._df[f'{cls._column}_mean'] = window.mean().shift()
        # cls._df[f'{cls._column}_var'] = window.var().shift()
        for q in cls._quantiles:
            cls._df[f'{cls._column}_quantile_{q}'] = window.quantile(q).shift()
