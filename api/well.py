import pandas as pd


class Well(object):

    def __init__(
            self,
            well_name: int,
            df: pd.DataFrame,
            rhoo: float,
    ):
        self.well_name = well_name
        self.df = df
        self.rhoo = rhoo
