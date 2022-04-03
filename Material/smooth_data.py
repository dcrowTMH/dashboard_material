import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

def smooth(s, n):
    # smooth the data with lowess
    if s.values[0] == 0:
        # filter the data if the first value is 0
        last_zero_date = s[s == 0].index[-1]
        s = s.loc[last_zero_date:]
        s_daily = s.diff().dropna()
    else:
        # first value not 0, use it to fill the first missing value
        s_daily = s.diff().fillna(s.iloc[0])
        
    # Dont smooth the data less than 15 values
    if len(s_daily) < 15:
        return s
    
    y = s_daily.values
    x = np.arange(len(y))
    # get the fraction for the lowess function (0 to 1)
    frac = n/len(y)
    y_lowess = lowess(y, x, frac = frac, to_sorted = True, return_sorted = False)
    # clip the data to ensure there is no negative value
    s_lowess = pd.Series(y_lowess, index = s_daily.index).clip(0)
    # get the cumlative data
    s_lowess_cumlative = s_lowess.cumsum()
    # get the last value for original/ smoothed data for direct alignment
    last_actual = s.values[-1]
    last_cumlative = s_lowess_cumlative.values[-1]
    s_lowess_cumlative *= last_actual/ last_cumlative
    return s_lowess_cumlative