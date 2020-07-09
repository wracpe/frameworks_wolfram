from itertools import product
from pandas import DataFrame, Series
from statsmodels.tsa.arima_model import ARIMA


class ArimaWrapper(object):

    p_range = range(0, 3)
    d_range = range(0, 2)
    q_range = range(0, 3)
    param_grid = [dict(p=x[0], d=x[1], q=x[2]) for x in product(p_range, d_range, q_range)]

    def __init__(self, p: int = 2, d: int = 1, q: int = 2):
        self.order = (p, d, q)
        self._model_fit = None

    def fit(self, x: DataFrame, y: Series):
        model = ARIMA(endog=y, order=self.order, exog=x, dates=x.index, freq='D')
        self._model_fit = model.fit(transparams=False, solver='powell', maxiter=1000, tol=1e-3, disp=False)

    def predict(self, x: DataFrame) -> Series:
        # y = self._model_fit.predict(start=x.index[0], end=x.index[-1], exog=x, typ='levels')
        y = self._model_fit.forecast(steps=len(x.index), exog=x)[0]
        y = Series(y, x.index)
        return y

    def set_params(self, **params):
        self.order = (params['p'], params['d'], params['q'])
