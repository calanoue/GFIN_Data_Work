"""
Format and clean data.
"""
import numpy as np
import numpy.ma as ma
from scipy.stats import linregress, mstats
from itertools import imap
from scipy import optimize

class CleanData:
    """
    Class to hold functions corresponding to cleaning and formatting of data.
    """
    def __init__(self, data, years, recents=7, perc_remove_1=.8, gap_length_to_fill=4,
                 outlier_p=98, outlier_t=False, max_neighbors=3, perc_remove_2=.6, perc_remove_3=.65):
        self.xs = data
        self.yrs = years
        self.rows_N = np.size(self.xs, 0)
        self.N = np.size(self.xs, 1)
        self.keep_n_values = recents # number of values at the end of the array to always keep
        self.xs_formats = np.zeros((self.rows_N, self.N), np.object) # array for holding formatting codes
        self.min_year = np.min(self.yrs) # minimum year
        self.init_perc_remove = perc_remove_1 # remove row if more masked values than this
        self.second_perc_remove = perc_remove_2 # remove row after outlier and average if greater than this
        self.perc_remove_after_start_idx = perc_remove_3 # remove row if after start index this many are masked
        self.max_gap_length = gap_length_to_fill # max gap length to fill with linear space
        self.outlier_perc = outlier_p # percentage for outlier score at percentile calculations
        self.outlier_normal = outlier_t # calculate outlier by percentile or normal dist
        self.masked_neighbors = max_neighbors # max number of masked neighbors for new start
        self.format_and_clean_data_main() # run main function to format
    
    def format_and_clean_data_main(self):
        """
        Main function to format and clean data based on choices by the user.
        """
        # Check if over missing_bound percent or missing_bound number of values are missing
        too_many_missing = self.has_too_many_missing(self.init_perc_remove)
        if ma.any(too_many_missing):
            idx, = ma.where(too_many_missing)
            self.xs[idx] = ma.mask_rows(self.xs[idx])

        # Check array to see if it is filled with values or empty
        if ma.all(self.check_for_all()):
            return self.xs, self.xs_formats

        # Clean outliers
        self.clean_outliers() # code ==> 1

        # Take average of neighbor values to fill up to a given missing value gap length
        self.clean_gaps_w_linspace(fill_gap_length=self.max_gap_length) #code ==> 2
        if ma.all(ma.count_masked(self.xs[:, :-self.keep_n_values], axis=1)[np.newaxis,:] == 0):
            return self.xs, self.xs_formats # if no masked values remain in values before recent ones

        # Remove values if they start the array and are then followed by too many masked values
        start_idx = self.find_new_starting_value() # code ==> 3

        # If there are over x% blank values left in the original data after above changes,
        # check to see if x% of the blanks fall after the new start year
        too_many_missing = self.has_too_many_missing(self.second_perc_remove) # boolean array
        if ma.any(too_many_missing):
            n_masked = np.array([ma.count_masked(self.xs[i,s_idx:])
                                 for i, s_idx in enumerate(start_idx)]) / self.N > self.perc_remove_after_start_idx
            if ma.any(n_masked):
                idx, = ma.where(n_masked)
                self.xs[idx] = ma.mask_rows(self.xs[idx])
                self.xs_formats[idx] = np.zeros(self.N)

        # To fill in remaining values, run linear regression on non-zero values
        self.clean_gaps_w_lin_regress(start_idx) # code ==> 4

        # If linear regression left negative or zero values,
        # then use linear space to fill in middle gaps
        if ma.any(ma.masked_less_equal(self.xs, 0.)):
            self.clean_gaps_w_linspace() # code ==> 5
    
    def check_for_all(self):
        """
        Function to check if all the values are masked or all not masked
        """
        masked_N = ma.count_masked(self.xs, axis=1)[np.newaxis,:]
        is_all_masked = self.N == masked_N
        is_none_masked = masked_N == 0
        return is_none_masked + is_all_masked
    
    def fill_xs_formats(self, idx, code):
        """
        Function to fill xs formatting array.

        Parameters
        ----------
        idx : numpy array
            Array of indices that need to be updated in the xs_formats array.
        code : integer
            Code corresponding to the change made. See code legend in class description.
        """
        w_idx = np.transpose(np.where(idx))
        for index in w_idx:
            row, col = index # row and column of index
            if self.xs_formats[row, col]: # if a format has already hit this row/col
                self.xs_formats[row, col] = np.array([self.xs_formats[row, col]] + [code])
            else:
                self.xs_formats[row, col] = code
    
    def clean_outliers(self):
        """
        Function to remove outliers.

        Parameters
        ----------
        self.outlier_perc : integer
            Percentile value for scipy.stats.scoreatpercentile function. Mask all values greater
            than this value.
        self.outlier_std_dev : integer
            If the data came from a normal distribution, the number of standard deviations
            from the mean for outlier consideration.
        self.outlier_normal : {False, True}
            Whether the data came from a normal distribution, for the type of outlier
            to calculate.
        """
        hldr = self.xs.copy()
        if self.outlier_normal: # outliers assuming normal distribution - num_rows * [min, max]
            std = self.outlier_std_dev * self.xs.std(1)
            outlier_all = np.reshape((self.xs.mean(1) + [-std, std]).T, (self.rows_N, 2))
        else: # outliers using percentiles - num_rows * [min, max]
            outlier_all = ma.array([[mstats.scoreatpercentile(self.xs[i, :], 100 - self.outlier_perc),
                   mstats.scoreatpercentile(self.xs[i, :], self.outlier_perc)] for i in xrange(self.rows_N)])
        self.xs = ma.array([ma.hstack((ma.masked_outside(self.xs[i, :-self.keep_n_values], outlier_all[i, 0],
            outlier_all[i, 1]), self.xs[i, -self.keep_n_values:])) for i in xrange(self.rows_N)])
        idx = ~hldr.mask * self.xs.mask # boolean array of changes
        if ma.any(idx): # if there have been any outlier removals
            self.fill_xs_formats(idx, 1)
    
    def clean_gaps_w_lin_regress(self, start_idx):
        """
        Function to clean gaps in the data with a linear regression.

        Parameters
        ----------
        start_idx : integer
            First non-masked value of array.
        """
        hldr = self.xs.copy() # copy of the data before changes
        non_zero_idx = np.transpose(self.xs.nonzero())
        for i in xrange(self.rows_N):
            idx = non_zero_idx[np.where(non_zero_idx[:, 0] == i)][:, 1]
            if idx.any():
                slope, intercept, r, p, se = linregress(self.yrs[idx], self.xs[i,idx])
                missing_xs = ma.where(self.xs[i, start_idx[i]:-self.keep_n_values].mask)[0] + start_idx[i]
                if np.any(missing_xs):
                    self.xs[i, missing_xs] = (self.min_year + missing_xs) * slope + intercept
        idx = hldr.mask * ~self.xs.mask
        if ma.any(idx): # if any values are filled with the linear regression
            self.fill_xs_formats(idx, 4)
    
    def clean_gaps_w_linspace(self, fill_gap_length=0):
        """
        Function to fill gaps with linearly spaced values.

        Parameters
        ----------

        fill_gap_length : integer
            Maximum length of gaps to be filled by averaging.
        """
        self.xs = ma.masked_less_equal(self.xs, 0.)
        hldr = self.xs.copy()
        condition = ma.getmask(self.xs)

        # Fill masked value gaps if not at the beginning or end of the array or if the
        # gap length is less than the max gap length
        cont_regions = self.contiguous_regions(condition)
        for step in np.arange(2, np.size(cont_regions, 0) + 1, 2): # 1st row start, 2nd stop
            (axis, blank), (start, stop) = cont_regions[step - 2:step].T
            gap_length = stop - start # length of gap
            if start and stop < self.N: # don't fill in gaps at beginning or end
                if not fill_gap_length or gap_length <= fill_gap_length:
                    self.xs[axis, start:stop] = np.linspace(self.xs[axis, start -1],
                        self.xs[axis, stop], stop - start + 1, endpoint=False)[1:]
        idx = hldr.mask * ~self.xs.mask
        if ma.any(idx): # if any values are filled with linear spacing
            self.fill_xs_formats(idx, 5)
    
    def find_new_starting_value(self):
        """
        Function to mask values at the beginning of the array if they have greater than
        max_masked_neighbors number of masked values after them.

        Parameters
        ----------
        self.masked_neighbors : integer
            Maximum number of masked values falling after a non-masked value.
        """
        hldr = self.xs.copy() # copy of the data before changes
        mask = ~self.xs.mask
        shift_coeffs = [-x for x in xrange(1, self.masked_neighbors + 1)]
        for shift in shift_coeffs:
            x_shifted = np.roll(self.xs, shift=shift, axis=1)
            mask *= ~x_shifted.mask
        start_idx = ma.argmax(mask, axis=1) # get first True or masked value for new starting year
        for i, s_idx in enumerate(start_idx):
            self.xs.mask[i, :s_idx] = True # mask all values before starting index
        idx = ~hldr.mask * self.xs.mask
        if ma.any(idx): # if eliminated any values due to falling before new starting year
            self.fill_xs_formats(idx, 3)
        return start_idx
    
    def contiguous_regions(self, condition):
        """
        Function to find contiguous True regions of the boolean array "condition".
        Returns a 2D array where the first column is the start index of the region and the
        second column is the end index.

        Parameters
        ----------
        condition : boolean
            Boolean array of the condition evaluations in two columns --> Start and stop.
        """
        diff = np.diff(condition) # Get changes in the condition
        idx = np.transpose(diff.nonzero()) # Find the indexes of changes in condition
        idx[:, 1] += 1 # Shift the column index by 1 to the right.
        if np.any(condition[:, 0][:, np.newaxis]):
            is_true_rows, = np.where(condition[:, 0]) # row with mask at start
            for i in is_true_rows: # append to end of the array
                idx = np.vstack((idx, [i, 0]))
        if np.any(condition[:, -1][:, np.newaxis]):
            is_true_rows, = np.where(condition[:, -1]) # row with mask at end
            for i in is_true_rows: # append to end of the array
                idx = np.vstack((idx, [i, np.size(condition, 1) - 1]))

        # Convert to recarray and then sort by indices, first by start and then stop
        idx = np.array([i for i in imap(tuple, idx)], dtype=[('start', int), ('stop', int)])
        idx = np.sort(idx, order=['start', 'stop']).view(int).reshape((-1, 2))
        return idx
    
    def has_too_many_missing(self, missing_bound):
        """
        Function to find if over missing_bound percent or
        missing_bound number of values are missing

        Parameters
        ----------
        missing_bound : if < 1 ==> float, otherwise integer
            Bound for checking if too many values are missing in the array.
        """
        masked_count = ma.count_masked(self.xs, axis=1) * 1.
        return [masked_count > missing_bound,
                masked_count / self.N > missing_bound][missing_bound < 1]
    
    def get_return_values(self):
        """
        Function to return the values and formats array.
        """
        return self.xs, self.xs_formats

class ExpSmooth:
    """Exponential trend smoothing class
    Example usage:
    from optimized_forecasting import ExpSmooth
    fCastPeriods = 20 # number of periods to forecast
    f = ExpSmooth() # initialize exponential smoothing class
    fitOpt = f.calc_variable_arrays(.98, xs, fCastPeriods) # initialize arrays once and run optimization routine
    xs = f.exp_smooth_forecast(fitOpt, True) # get exp smooth forecasted
    """
    def __init__(self):
        self.x0 = (0.25, 0.50, .98) # (trendWeight, levelWeight, trendModifier) - starting values
        self.constr1 = lambda x0: .98 - x0[2]  # Constraint 1: trendModifier < .98
        self.constr2 = lambda x0: x0[2] - .98  # Constraint 2: trendModifier > .98
        self.constr3 = lambda x0: .95 - x0[1]  # Constraint 3: lvlWeight < .95
        self.constr4 = lambda x0: x0[1] - .05  # Constraint 4: lvlWeight > .05
        self.constr5 = lambda x0: .25 - x0[0]  # Constraint 5: trendWeight < .25
        self.constr6 = lambda x0: x0[0] - 0.   # Constraint 6: trendWeight > 0.
        self.constrs = [self.constr1, self.constr2, self.constr3,
                        self.constr4, self.constr5, self.constr6
        ]

    def calc_variable_arrays(self, trend_modifier, data, periods):
        """
        Function to calculate arrays that depend on changing starting variables, i.e. trend modifier

        Parameters
        ----------
        trend_modifier : float
            Trend modifier value for exponential dampened smoothing function.
        data : array
            Data array that needs to be forecasted
        periods : integer
            Number of periods to forecast.
        """
        self.xs = data #data
        self.fcast_periods = periods # number of periods to forecast
        self.data_N = self.xs.size # count of data
        self.warmUp = self.data_N / 2 # going with a warm-up of half the data, instead of 6
        self.zero_len = self.data_N + self.fcast_periods # length of arrays with one missing value
        self.lvlArr = np.zeros(self.zero_len + 1)
        self.errArr = np.zeros(self.data_N)
        self.trendArr = np.zeros(self.zero_len + 1)
        self.fcasts = np.zeros(self.zero_len)
        avgFirstFourDiffs = np.average(np.diff(self.xs[:4 + 1])) # average difference of first four data points
        self.trendArr[0] = avgFirstFourDiffs
        self.lvlArr[0] = self.xs[0] - avgFirstFourDiffs
        self.trendMod = trend_modifier
        self.trendExp = np.ones(self.zero_len + 1) #trend exponent array
        self.trendExp[self.data_N:] = np.power(
            self.trendMod, (np.arange(self.data_N + 1, self.zero_len + 2) - self.data_N)
        )
        self.trendMult = np.zeros(self.zero_len + 1) #trend multiplier array
        self.trendMult[self.data_N:] = np.cumsum(self.trendExp[self.data_N:])
        return optimize.fmin_cobyla(self.exp_smooth_forecast, self.x0, self.constrs, iprint=0)

    def exp_smooth_forecast(self, x0, final=False):
        """
        Function for exponential smoothing forecast routine

        Parameters
        ----------
        x0 : array
            Array of initial trend variables [trend_weight, level_weight, trend_modifier]
        final : {False, True}
            If final return forecasted array, otherwise use MSE for minimizing the function.
        """
        self.trendWeight, self.lvlWeight, self.trendMod = x0 # initial trend variables
        for i in xrange(1, self.data_N + 1):
            tmp = self.lvlArr[i - 1] + self.trendArr[i - 1]*self.trendMod
            self.fcasts[i - 1] = (not tmp < 0)*tmp
            self.errArr[i - 1] = self.xs[i - 1] - self.fcasts[i - 1]
            self.trendArr[i] = self.trendMod*self.trendArr[i - 1] + self.trendWeight*self.errArr[i - 1]
            self.lvlArr[i] = self.fcasts[i - 1] + self.lvlWeight*self.errArr[i - 1]

        # Return forecast array concatenated with original data array, if final
        # else return forecasting MSE for the purpose of minimizing the function
        if final:
            self.lvlArr[self.data_N + 1:] = self.lvlArr[self.data_N]
            self.trendArr[self.data_N + 1:] = self.trendArr[self.data_N]
            self.fcasts[self.data_N - 1:] = self.lvlArr[self.data_N:] +\
                                            self.trendArr[self.data_N:] * self.trendMult[self.data_N:]
            self.fcasts = np.ma.masked_less_equal(self.fcasts, 0)
            return np.ma.hstack((self.xs[:self.data_N], self.fcasts[self.data_N:]))
        else:
            sumAbsErrs2 = np.cumsum(self.errArr**2)
            warm_up_SSE = sumAbsErrs2[self.warmUp - 1]
            fcast_SSE = sumAbsErrs2[self.data_N - 1]
            SSE_diff = fcast_SSE - warm_up_SSE
            return SSE_diff / self.fcast_periods

class MiscForecastFunctions:
    """
    Miscellaneous forecasting functions:
    --> Polynomial trend - default is a one-degree polynomial aka a linear regression
    """
    def linear_regression(self, t, xs, periods, deg=1):
        """
        Function to forecast with a linear trend.
        Example Usage:
        from optimized_forecasting import MiscForecastFunctions
        misc = MiscForecastFunctions()
        xs = misc.linear_regression(data['yr'].view(int), data['valFld'].view(float), fcast_periods)
        """
        #TODO - change to expected value calculations to calculate multiple arrays at once
        slope = ((X * Y).mean(axis=1) - X.mean() * Y.mean(axis=1)) / ((X**2).mean() - (X.mean())**2) / Y[:, 0]
        intercept = Y.mean(axis=1) - slope*X.mean()
        max_t = np.max(t) + 1 # maximum time periods plus one for np.arange function
        fcast_periods = np.arange(max_t, periods + max_t) # periods to forecast
        fcasts = polyval(z, fcast_periods) # evaluate the polynomial at the forecast years
        return np.hstack((xs, fcasts))

    def cons_forecast_multiple(self, gdp, cons, elast, periods, n, last_val_idx):
        """
        Forecast consumption with multiple sets of data - faster than one at a time.
        Last value index must be the same across all countries for this function to work.
        arguments --> GDP data, consumption data, elasticity data, number of years to forecast,
        number of data columns, and index of last actual non-zero data point.
        return --> original data concatenated w/ forecasted data
        Example usage (with the cons, gdp, and elasticity data extracted from the database):
        from optimized_forecasting import MiscForecastFunctions
        misc = MiscForecastFunctions()
        n = len(app.BarsToPlot) #number of data columns, i.e. countries
        last_val_idx = np.max(np.reshape(np.nonzero(cons_data)[1], (n, -1)), axis=1)
        all_equal = np.all(last_val_idx==last_val_idx[0], axis=0) #check if indices are all same
        if all_equal:
            xs = misc.cons_forecast_multiple(gdp_data, cons_data, elast_data,
                2030 - 1961 - last_val_idx, n, last_val_idx)
        """
        gdp_data = gdp.reshape(n, -1)
        cons_data = cons.reshape(n, -1)
        elast_data = elast.reshape(n, -1)
        fill = np.zeros((n, periods))
        d_ln = np.exp(elast_data * np.diff(np.log(gdp_data[:, last_val_idx:])))
        fill[:, 0] = cons_data[:, last_val_idx] * d_ln[:, 0] #calculate the first value
        for i in xrange(n):
            for j in xrange(1, periods): # need a for loop due to recursion problems with numpy
                fill[i, j] = fill[i, j - 1] * d_ln[i, j]
        return np.hstack((cons_data[:, :last_val_idx + 1], fill))

# Get data from database for formatting and then forecast

import sqlite3
from optimized_forecasting import ExpSmooth

# Connect to the database and create a cursor for extracting data
DB = r".\GFIN_DB.db3"
connection = sqlite3.connect(DB)
cursor = connection.cursor()

# Globals
last_year = 2011
YRS = np.arange(1961, last_year) # years for commodity values
forecast_year = 2030 # forecast out to this year
flds = ",".join("yr%s"%i for i in YRS)
Q = "SELECT %s FROM Commodity_Raw_Data LIMIT 1000"%flds

# Clean data according to user requests
xs = ma.masked_less(np.array(cursor.execute(Q).fetchall()), 0)
values = CleanData(xs, YRS)
x, x_formats = values.get_return_values()
ks = np.transpose(ma.notmasked_edges(x, axis=1)) # min and max non-zero values
f = ExpSmooth() # initialize exponential smoothing class
N = YRS.size
for i, k in enumerate(ks): # forecast each array
    start, stop = k[1, :]
    periods = forecast_year - last_year + (N - stop)
    xs = x[i, start:stop + 1]
    fitOpt = f.calc_variable_arrays(.98, xs, periods) # initialize arrays once and run optimization routine
    xs = ma.hstack((ma.masked_all(start), f.exp_smooth_forecast(fitOpt, True)))

# Close cursor and connection
cursor.close()
connection.close()