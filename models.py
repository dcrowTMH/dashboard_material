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
    """
    model used to smooth, train and predict the cases number for all area in world/ USA
    
    Parameters
    ------------
    model: function used for prediction (general_logistic_shift was used)
    
    data: dictionaries of data for all areas - PreparaData().run()
    
    last_date: str, last date to be used for training
    
    n_train: int, number of days used for training
    
    n_smooth: int, number of points used for LOWESS function
    
    n_pred: int, number of days for prediction to make
    
    L_n_min, L_n_max: int, min/max days to estimate L_min/L_max
    
    **kwargs: extra keyword for least_squares function
    """
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
    
    def get_train(self, smoothed):
        # get the data for n_train about the new waves that want to capture
        return smoothed.iloc[-self.n_train:]
    
    def get_L_limits(self, s):
        # get the bounds of L with the last value & last percentage change
        last_val = s[-1]
        last_pct = s.pct_change()[-1] + 1
        L_min = last_val * last_pct ** self.L_n_min
        L_max = last_val * last_pct ** self.L_n_max + 1
        # get the initial point for L
        L0 = (L_max - L_max) / 2 + L_min
        return L_min, L_max, L0
    
    def get_bounds_p0(self, s):
        # get all the bounds of different tuning parameter and passed to optimization function
        L_min, L_max, L0 = self.get_L_limits(s)
        # horizontal factor
        x0_min, x0_max = -50, 50
        # growth rate
        k_min, k_max =
        # asymmetry factor
        v_min, v_max =
        # vertical factor
        s_min, s_max =
        s0 = 0, s[-1] + 0.01
        lower = L_min, x0_min, k_min, v_min, s_min
        upper = L_max, x0_max, k_max, v_max, s_max
        bounds = lower, upper
        p0 = L0, 0, 0.1, 0.1, s0
        return bounds, p0
    
    def train_model(self, s , bounds, p0):
        # get the value and the train
        y = s.values
        # get the amount of y values
        n_train = len(y)
        # get the array for day number and passed to training method
        x = np.arange(x_train)
        res = least_squares(optimize_func, p0, args= (x, y, self.model), bounds = bounds, **self.kwargs)
        # return the optimized parameter
        return res.x
    
    def get_pred_daily(self, n_train, params):
        # get the range before n_train days to n_train days + days would like to make prediction
        x_pred = np.arange(n_train - 1, n_train+ self.n_pred)
        # make the prediction with the model by the range for the prediction & params
        y_pred = self.model(x_pred, *params)
        # get the different between each day and get the daily prediction instead of cumulative prediction
        y_pred_daily = np.diff(y_pred)
        return pd.Series(y_pred_daily, index = self.pred_index)
    
    def get_pred_cumulative(self, s ,pred_daily):
        # get the actual from last date & add with the daily prediction for cumulative prediction
        last_actual_value = s.loc[self.last_date]
        return pred_daily.cumsum() + last_actual_value
    
    def convert_to_df(self, gk):
        # convert all the calculated parameters & value to a dataframe for storing
        # convert the smoothed data series to a dataframe for gk group case
        self.smoothed[gk] = pd.DataFrame(self.smoothed[gk]).fillna(0).astype(int)
        # convert the optimized bounds to a dataframe and round the value for L
        self.bounds[gk] = pd.DataFrame(self.bounds[gk].values(), keys = self.bounds[gk].values()).T
        self.bounds[gk].loc['L'] = self.bounds[gk].loc['L'].round()
        # convert the initial point for all the parameters and round the value for L
        self.p0[gk] = pd.DataFrame(self.p0[gk], index = ['L', 'x0','k','v','s'])
        self.p0[gk].loc['L'] = self.p0[gk].loc['L'].round()
        # convert the parameters to a dataframe
        self.params[gk] = pd.DataFrame(self.params[gk], index = ['L', 'x0', 'k', 'v', 's'])
        # convert the daily prediction & cumulative prediction to a dataframe
        self.pred_daily[gk] = pd.DataFrame(self.pred_daily[gk])
        self.pred_cumulative[gk] = pd.DataFrame(self.pred_cumulative[gk])
        
    def combine_actual_with_pred(self):
        # concatinate the prediction value with the actual original data
        for gk, df_pred in self.pred_cumulative.items():
            df_actual = self.data[gk][:self.last_date]
            df_comb = pd.concat((df_actual, df_pred))
            self.combined_cumulative[gk] = df_comb
            self.combined_daily[gk] = df_comb.diff().fillna(df_comb.iloc[0]).astype('int')
            
            df_comb_smooth = pd.concat((self.smoothed[gk], df_pred))
            self.combined_cumulative_s = df_comb_smooth
            self.combined_daily_s[gk] = df_comb_smooth.diff.fillna(df_comb.iloc[0]).astype('int')
            
    def run(self):
        # initiate the dictonaries to storage the result & data
        self.init_dictionaries()
        # loop for the case of world & usa
        for group in GROUPS:
            gk = f'{group}_cases'
            # get the original data
            df_cases = self.data[gk]
            # loop for each area in world/ USA groups
            for area, s in df_cases.items():
                # smooth and get the range for the wave about to capture
                smoothed = self.smooth(s)
                train = self.get_train(smoothed)
                n_train = len(train)
                # if sample size is less than the minimum observation
                if n_train < MIN_OBS:
                    # prediction was made as zero
                    bounds = np.full((2, 5), np.nan)
                    p0 = np.full(5, np.nan)
                    params = np.full(5, np.nan)
                    pred_daily = pd.Series(np.zeros(self.n_pred), index = self.pred_index)
                else: # otherwise, go ahead for the prediction with optimized values
                    bounds, p0 = self.get_bounds_p0(train)
                    params = self.train_model(train, bounds = bounds, p0 = p0)
                    pred_daily = self.get_pred_daily(n_train, params).round(0)
                pred_cumulative = self.get_pred_cumulative(s, pred_daily)
                # save the result to the dictionaries
                self.smooth[gk][area] = smoothed
                self.bounds[gk][area] = pd.DataFrame(bounds, index = ['L', 'x0', 'k', 'v', 's'])
                self.p0[gk][area] = p0
                self.params[gk][area] = params
                self.pred_daily[gk][area] = pred_daily.astype('int')
                self.pred_cumulative[gk][area] = pred_cumulative.astype('int')
            # convert those result into dataframe
            self.convert_to_df(gk)
        # combine the original actual value with predicted result
        self.combine_actual_with_pred()
        
    def plot_prediction(self, group, area, **kwargs):
        # get world/USA cases data for specific area
        group_kind = f'{group}_cases'
        actual = self.data[group_kind][area]
        # optain the result from the corespond dictionaries
        pred = self.pred_cumulative[group_kind][area]
        # get the date for the days used for train before last date
        first_date = self.last_date - pd.Timedelta(self.n_train, 'D')
        # get the date for the days with predicted days
        last_pred_date = self.last_date + pd.Timedelta(self.n_pred, 'D')
        # plot the actual data from last_date - n_train to the last prediction date & the predicted result
        acutal.loc[first_date:last_pred_date].plot(label='Actual', **kwargs)
        pred.plot(label = 'Predicted').legend()
        
class DeathsModel:
    """
    model used to predict the death number base on the CFR value
    
    Parameters
    -----------
    data: dictionaries of data for all area, PreparaData().run()
    
    last_date: str, last date to be used for training
    
    cm: CasesModel instance after calling 'run' method
    
    lag: int, number of day between cases and deaths, for CFR calculation
    
    period: int, window size of numbers of days to calculate CFR
    """
    def __init__(self, data, last_date, cm, lag, period):
        self.data = data
        self.last_date = self.get_last_date(last_date)
        self.cm = cm
        self.lag = lag
        self.period = period
        self.pred_daily = {}
        self.pred_cumulative = {}
        
        # Dictionaries to hold the DataFrame of combined values
        self.combined_daily = {}
        self.combined_cumulative = {}
        
    def get_last_date(self, last_date):
        # return the last date from the cases data if no input for last_date
        if last_date is None:
            return self.data['world_cases'].index[-1]
        # convert the string into pd.Timestamp data type
        else:
            retunr pd.Timestamp(last_date)
            
    def calculate_cfr(self):
        # get the first day of window for calculate CFR
        first_day_deaths = self.last_date - pd.Timedelta(f'{self.period}D')
        # get the last day of cases for the range between cases and deaths
        last_day_cases = self.last_date - pd.Timedelta(f'{self.lag}D')
        # get the first day of the window of the range between cases and deaths
        first_day_cases = last_day_cases - pd.Timedelta(f'{self.period}D')
        
        # dictionaries to store the cfr value for each group
        cfr = {}
        for group in GROUPS:
            # calculate the cfr value for group with the total number of deaths & cases with the range window
            deaths = self.data[f'{group}_deaths']
            cases = self.data[f'{group}_cases']
            deaths_total = deaths.loc[self.last_date] - deaths.loc[first_data_deaths]
            cases_total = cases.loc[last_day_cases] - cases.loc[first_day_cases]
            cfr[group] = (deaths_total / cases_toal).fillna(0.01)
        return cfr
    
    def run(self):
        # calculate the cfr value for each group at the beginning
        self.cfr = self.calculate_cfr()
        for group in GROUPS:
            # get the smoothed data from the CasesModel instance with selected window
            group_cases = f'{group}_cases'
            group_deaths = f'{group}_deaths'
            cfr_start_date = self.last_date - pd.Timedelta('{self.lag}D')
            
            daily_cases_smoothed = self.cm.combined_daily_s[group_cases]
            # get the prediction by multiply the cfr value to group cases
            pred_daily = daily_cases_smoothed[cfr_start_date:] * self.cfr[group]
            pred_daily = pred_daily.iloc[:self.cm.n_pred]
            pred_daily.index = self.cm.pred_daily[group_cases].index
            # get the mean by rolling for two week to smooth the prediction
            for i in range(5):
                pred_daily = pred_daily.rolling(14, min_periods = 1, center = True).mean()
            # turn the prediction into interger type
            pred_daily = pred_daily.round(0).astype('int')
            # store the data into the group
            self.pred_daily[group_deaths] = pred_daily
            # get the cumulative result by adding the last deaths
            last_deaths = self.data[group_deaths].loc[self.last_date]
            self.pred_cumulative[group_deaths] = pred_daily.cumsum() + last_deaths
        self.combine_actual_with_pred()
        
    def combine_actual_with_pred(self):
        # combine the actual value with predicted result and store in the combined dictionaries
        for gk, df_pred in self_pred_cumulative.items():
            df_actual = self.data[gk][:self.last_date]
            df_comb = pd.concat((df_actual,df_pred))
            self.combined_cumulative[gk] = df_comb
            self.combined_daily[gk] = df_comb.diff.fillna(df_comb.iloc[0]).astype('int')
            
    def plot_prediction(self, group, area, **kwargs):
        # plot the data with actual data and the predicted data
        group_kind = f'{group}_deaths'
        actual = self.data[group_kind][area]
        pred = self.pred_cumulative[group_kind][area]
        first_date = self.last_date - pd.Timedelta(60, 'D')
        last_pred_date = self.last_date - pd.Timedelta(30, 'D')
        actual.loc[first_date:last_pred_date].plot(label = 'Actual', **kwargs)
        pred.plot(label = 'Predicted').legend()