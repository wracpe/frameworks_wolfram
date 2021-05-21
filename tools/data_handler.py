import pathlib
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from typing import Dict

from config_field import *
from tools.window_feature_generator import WindowFeatureGenerator


class DataHandler(object):

    def __init__(
            self,
            path_data: pathlib.Path,
            target: str,
            window_sizes: List[int],
            quantiles: List[float],
    ):
        self.path = path_data
        self.target = target
        self.window_sizes = window_sizes
        self.quantiles = quantiles


def read_prepare() -> Dict[str, Dict]:
    df = _read_chess()

    wells = df['Скв'].unique().tolist()
    well_data = dict()

    for well in wells:
        df_well = df[df['Скв'] == well]
        df_well = _fill_missing_values(df_well, predictor)
        df_well = _fill_missing_values(df_well, predicate)
        df_well = df_well[usable_columns]
        df_well.set_index(keys='Дата', inplace=True, verify_integrity=True)

        # df_1 = df_well.diff()
        # df_2 = df_well.shift()
        # df_well = df_1.divide(df_2)
        # df_well.dropna(inplace=True)
        df_well = WindowFeatureGenerator.run(df_well)

        data = _split_df_well(df_well)
        data['start_row'] = df_well.iloc[[0]]
        well_data[well] = data
        print(well)
    
    return well_data


def _read_chess() -> pd.DataFrame:
    df = pd.read_csv(filepath_or_buffer=data_path / field / 'град_штр.csv',
                     sep=';',
                     dtype={0: str},
                     parse_dates=[1],
                     dayfirst=True,
                     encoding='windows-1251')
    return df


def _fill_missing_values(df_well: pd.DataFrame, target: str) -> pd.DataFrame:
    estimator = ExtraTreesRegressor(n_jobs=-1, random_state=1)
    iter_imp = IterativeImputer(estimator,
                                max_iter=100,
                                tol=0.01,
                                initial_strategy='median',
                                imputation_order='descending',
                                skip_complete=True,
                                random_state=1)

    df = _select_features(df_well, target)
    x = iter_imp.fit_transform(df)
    df = pd.DataFrame(x, index=df.index, columns=df.columns)
    df_well[target] = df[target]
    return df_well


def _select_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    s_target = df[target]
    df.drop(columns=[target], inplace=True)
    df.dropna(axis='columns', how='all', inplace=True)

    num = df.count()
    corr = abs(df.corrwith(s_target, drop=True, method='pearson'))
    rank = pd.concat(objs=[num, corr], axis='columns')
    rank.columns = ['num', 'corr']
    rank.dropna(axis='index', how='any', inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(rank)
    rank = pd.DataFrame(data=x, index=rank.index, columns=rank.columns)

    rank = rank['num'] + rank['corr']
    rank.sort_values(ascending=False, inplace=True)
    rank = rank.head(features_num)

    features = list(rank.index)
    df = df[features]
    df.insert(loc=0, column=target, value=s_target.array)
    return df


def _split_df_well(df_well: pd.DataFrame):
    data = dict()
    total_samples_number = len(df_well.index)
    df_train = df_well.head(total_samples_number - forecast_days_number)
    df_test = df_well.tail(forecast_days_number)
    data['x_train'], data['y_train'] = _divide_x_y(df_train)
    data['x_test'], data['y_test'] = _divide_x_y(df_test)
    return data


def _divide_x_y(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    x = df.drop(columns=predicate)
    y = df[predicate]
    return x, y
