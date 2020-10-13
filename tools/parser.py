import pandas as pd

import settings as sett

from pandas import DataFrame
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List

import matplotlib.pyplot as plt

from tools.window_feature_generator import WindowFeatureGenerator


class Parser(object):

    _df: DataFrame
    _df_well: DataFrame
    _start_row: DataFrame
    _well_data: Dict[str, Dict]
    _well_names: List[str]

    @classmethod
    def parse(cls) -> Dict[str, Dict]:
        cls._read_df()
        cls._fill_missing_values_in_predictor()
        cls._prepare_train_test_data_by_wells()
        return cls._well_data

    @classmethod
    def _read_df(cls):
        cls._df = pd.read_csv(filepath_or_buffer=sett.work_path / sett.file_name,
                              sep=';',
                              header=0,
                              dtype={'A': str},
                              parse_dates=[1],
                              dayfirst=True,
                              encoding='windows-1251')

        cls._well_names = cls._df['Скв'].unique().tolist()

        # TODO: Rows 39-42 only for Otdelnoe filed calculation.
        #  Delete or change these rows for calculation other filed.
        cls._well_names.remove('7')
        cls._well_names.remove('7Г')
        cls._well_names.remove('7Г2')
        # cls._well_names.remove('55')
        # cls._well_names.remove('32')
        cls._well_names = ['22']

    @classmethod
    def _fill_missing_values_in_predictor(cls):
        imp_mean = IterativeImputer(max_iter=1000, initial_strategy='median', random_state=1)
        features = ['Давление забойное от Pпр',
                    'Давление на приеме насоса',
                    'Давление забойное от Hд',
                    #'Дебит нефти (ТМ)',
                    #'Дебит жидкости (ТМ)',
                    #'Дебит газа попутного',
                    'Удельный расход электроэнергии с контроллера']

        x = cls._df[features].copy()
        x = imp_mean.fit_transform(x)
        cls._df[features] = x

    @classmethod
    def _prepare_train_test_data_by_wells(cls):
        cls._well_data = dict()
        for well_name in cls._well_names:
            cls._df_well = cls._df[cls._df['Скв'] == well_name]
            # cls._df_well.drop(columns='Скв', inplace=True)
            # cls._df_well.dropna(axis='columns', how='all', inplace=True)
            # cls._df_well.set_index(keys='Дата', inplace=True, verify_integrity=True)
            #
            # s_target = cls._df_well[sett.predicate]
            # scaler = MinMaxScaler(feature_range=(0.1, 1))
            # s_corr = abs(cls._df_well.corrwith(s_target, drop=False, method='pearson'))
            # s_corr.dropna(inplace=True)
            # s_num = cls._df_well.count()
            # s_num = s_num.loc[s_corr.index]
            #
            # df_features = pd.concat(objs=[s_corr, s_num], axis='columns')
            # df_features.drop(index=sett.predicate, inplace=True)
            # X = scaler.fit_transform(df_features)
            # df_features = pd.DataFrame(data=X, index=df_features.index)
            #
            # features = df_features[0].mul(df_features[1])
            # features = features[features > 0.8]
            # features.sort_values(ascending=False, inplace=True)
            # features = list(features.index)
            # features.append(sett.predicate)
            #
            # df_f_well = cls._df_well[features]
            # df_f_well = cls._fill_missing_values_in_predictor(df_f_well)
            #
            # df_f_well.plot()
            # plt.show()
            #
            cls._df_well = cls._df_well[sett.usable_columns]
            # cls._df_well[sett.predicate] = df_f_well[sett.predicate]
            cls._prepare_df_well()

            # cls._df_well = cls._df_well.tail(395 + 90)

            data = cls._split_df_well()
            data['start_row'] = cls._start_row
            cls._well_data[well_name] = data

    @classmethod
    def _prepare_df_well(cls):
        cls._df_well.set_index(keys='Дата', inplace=True, verify_integrity=True)
        cls._df_well = cls._df_well.astype(dtype='float64')
        cls._df_well.interpolate(method='linear', axis='index', inplace=True, limit_direction='forward')
        cls._df_well.dropna(inplace=True)

        cls._start_row = cls._df_well.iloc[[0]]

        df_1 = cls._df_well.diff()
        df_2 = cls._df_well.shift()
        cls._df_well = df_1.divide(df_2)
        cls._df_well.dropna(inplace=True)

        cls._df_well = WindowFeatureGenerator.run(cls._df_well)

        # target_series = cls._df_well[sett.predicate]
        # feature_correlation_series = abs(cls._df_well.corrwith(target_series, drop=True, method='pearson'))
        # feature_correlation_series.sort_values(ascending=False, inplace=True)
        # pass

    @classmethod
    def _split_df_well(cls):
        data = dict()
        total_samples_number = len(cls._df_well.index)

        # df = cls._df_well.head(total_samples_number - 90)
        # cls._df_well = df.copy()
        #
        # total_samples_number = len(cls._df_well.index)

        df_train = cls._df_well.head(total_samples_number - sett.forecast_days_number)
        df_test = cls._df_well.tail(sett.forecast_days_number)
        data['x_train'], data['y_train'] = cls._divide_x_y(df_train)
        data['x_test'], data['y_test'] = cls._divide_x_y(df_test)
        return data

    @staticmethod
    def _divide_x_y(df):
        x = df.drop(columns=sett.predicate)
        y = df[sett.predicate]
        return x, y
