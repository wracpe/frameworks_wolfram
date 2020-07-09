import pandas as pd

import settings as sett

from pandas import DataFrame
from typing import Dict

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class Parser(object):

    _df: DataFrame
    _dfs: Dict[str, DataFrame]

    @classmethod
    def parse(cls) -> Dict[str, DataFrame]:
        cls._read_df()
        cls._split_df_by_wells()
        return cls._dfs

    @classmethod
    def _read_df(cls):
        cls._df = pd.read_csv(filepath_or_buffer=sett.work_path / sett.file_name,
                              sep=';',
                              header=0,
                              dtype={'A': str},
                              parse_dates=[1],
                              dayfirst=True,
                              encoding='windows-1251')

    @classmethod
    def _split_df_by_wells(cls):
        well_names = cls._df['Скв'].unique().tolist()

        # TODO: Rows 31-33 only for Otdelnoe filed calculation.
        #  Delete or change these rows for calculation other filed.
        well_names.remove('7')
        well_names.remove('7Г')
        well_names.remove('7Г2')
        well_names.remove('68Р')

        # well_names = ['4']

        imp_mean = IterativeImputer(max_iter=1000, random_state=0)
        features = ['Давление забойное от Pпр',
                    'Давление на приеме насоса',
                    'Давление забойное от Hд',
                    'Давление затрубное (ТМ)\t']

        x = cls._df[features].copy()
        x = imp_mean.fit_transform(x)
        cls._df[features] = x

        cls._dfs = dict.fromkeys(well_names)
        for well_name in well_names:
            df_well = cls._df[cls._df['Скв'] == well_name]
            df_well = df_well[sett.usable_columns]
            df_well = cls._prepare_df(df_well)
            cls._dfs[well_name] = df_well

    @staticmethod
    def _prepare_df(df: DataFrame) -> DataFrame:
        df.set_index(keys='Дата', inplace=True, verify_integrity=True)
        df = df.astype(dtype='float64')
        df.interpolate(method='linear', axis='index', inplace=True, limit_direction='forward')
        df.dropna(axis='index', how='any', inplace=True)
        return df
