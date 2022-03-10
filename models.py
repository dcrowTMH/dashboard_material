import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from statsmodel.nonparametric.smoothers_lowess import lowess
plt.style.use('dashboard.mplstyle')

GROUPS = 'world', 'usa'
KINDS = 'cases', 'deaths'
MIN_OBS = 15 # Minimum oberservations

def general_logistic_shift(x, L, x0, k, v, s):
    return (L - s) / ((1 + np.exp(-k * (x - x0))) **(1/v)) + s

def optimize_func(params, x, y, model):
    y_pred = model(x, params)
    error = y - y_pred
    return error

class CasesModel:
    def __init__(self, model, data, last_date, n_train, n_smooth, n_pred, L_n_min, L_n_max, **kwarg):
        """
        """
        self.model = model
        self.data = data
        self.last_date = self.get_last_date(last_date)
        self.n_train = self.n_train
        self.n_smooth = self.n_smooth
        self.n_pred = self.n_pred
        self.L_n_min = L_n_min
        self.L_n_max = L_n_max
        sefl.kwarg = kwarg
        
    def get_last_date(self, last_date):
        if last_date is None:
            return self.data['world_cases'].index[-1]
        else:
            return pd.Timestamp(last_date)
        
    def init_dictionaries(self):
        # create dictionaries to store the result for each group & kind at the beginning
        self.smoothed = {'world_cases' : {}, 'usa_cases' : {}}
        self.bounds = {'world_cases' : {}, 'usa_cases' : {}}
        self.p0 = {'world_cases' : {}, 'usa_cases' : {}}
        self.params = {'world_cases' : {}, 'usa_cases' : {}}
        self.pre_daily = {'world_cases' : {}, 'usa_cases' : {}}
        self.pre_cumulative = {'world_cases' : {}, 'usa_cases' : {}}
        
        # create the dictionaries to store the original & predicted data
        self.combined_daily = {}
        self.combined_cumulative = {}
        
        # create the dictionaries same as above but with smoothed data
        self.combined_daily_s = {}
        self.combined_cumulative_s = {}
        
    def smooth(self, s):
        s = s[:self.last_date]
        if s.values[0] == 0:
            # find the last zero date if the first value equal to zero
            last_zero_date = s[s == 0].index[-1]
            # get rid of all zero and find the different between the date
            s = s.loc[last_zero_date:]
            s_daily = s.diff().dropna()
        else:
            # first value not equal zero and find the different between the date
            s_daily = s.diff().fillna(s.iloc[0])
            
        # don't smooth the data if data amount is less than minimum observation
        if len(s_daily) < MIN_OBS:
            return s_daily.cumsum()
        
        y = s_daily.values
        # perform the lowess with day 1 to day n
        x = np.arange(len(y))
        # get the fraction from the smooth number from input
        frac = self.n_smooth / len(y) # decide the window size
        y_pred = lowess(y, x, frac = frac, is_sorted = True, return_sorted = False)
        s_pred = pd.Series(y_pred, index = s_daily.index).clip(0)
        s_pred_cumulative = s_pred.cumsum()
        
        if s_pred_cumulative[-1] == 0:
            # get back to original data if value was smoothed to zero
            return s_daily.cumsum()
        
        last_actual = s.values[-1]
        last_smoothed = s_pred_cumulative.values[-1]
        s_pred_cumulative *= last_acutal / last_smoothed
        return s_pred_cumulative