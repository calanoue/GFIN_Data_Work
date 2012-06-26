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
    def __init__(self, data, years, recents=7, perc_remove_1=.8, gap_length_to_fill=6,
                 outlier_p=98, max_neighbors=3, perc_remove_2=.6, perc_remove_3=.65):
        self.xs = data
        self.yrs = years
        self.rows_N = self.xs.shape[0]
        self.N = self.xs.shape[1]
        self.keep_n_values = recents # Number of values at the end of the array to always keep
        self.min_year = np.min(self.yrs)
        self.init_perc_remove = perc_remove_1 # Remove row if more masked values than this
        self.second_perc_remove = perc_remove_2 # Remove row after outlier and average if greater than this
        self.perc_remove_after_start_idx = perc_remove_3 # Remove row if after start index this many are masked
        self.max_gap_length = gap_length_to_fill # Max gap length to fill with linear space
        self.outlier_perc = outlier_p # Percentage for outlier score at percentile calculations
        self.masked_neighbors = max_neighbors # Max number of masked neighbors for new start
        self.format_and_clean_data_main() # Run main function to format
    
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
            return self.xs

        # Clean outliers
        self.clean_outliers()

        # Take average of neighbor values to fill up to a given missing value gap length
        self.clean_gaps_w_linspace(fill_gap_length=self.max_gap_length)
        if ma.all(ma.count_masked(self.xs[:, :-self.keep_n_values], axis=1)[np.newaxis,:] == 0):
            return self.xs # if no masked values remain in values before recent ones

        # Remove values if they start the array and are then followed by too many masked values
        start_idx = self.find_new_starting_value()

        # If there are over x% blank values left in the original data after above changes,
        # check to see if x% of the blanks fall after the new start year
        too_many_missing = self.has_too_many_missing(self.second_perc_remove) # boolean array
        if ma.any(too_many_missing):
            n_masked = np.array([ma.count_masked(self.xs[i,s_idx:])
                                 for i, s_idx in enumerate(start_idx)]) / self.N > self.perc_remove_after_start_idx
            if ma.any(n_masked):
                idx, = ma.where(n_masked)
                self.xs[idx] = ma.mask_rows(self.xs[idx])

        # To fill in remaining values, run linear regression on non-zero values
        self.clean_gaps_w_lin_regress(start_idx)

        # If linear regression left negative or zero values, then use linear space to fill in middle gaps
        if ma.any(ma.masked_less_equal(self.xs, 0.)):
            self.clean_gaps_w_linspace()
    
    def check_for_all(self):
        """
        Function to check if all the values are masked or all not masked
        """
        masked_N = ma.count_masked(self.xs, axis=1)[np.newaxis,:]
        is_all_masked = self.N == masked_N
        is_none_masked = masked_N == 0
        return is_none_masked + is_all_masked
    
    def clean_outliers(self):
        """
        Function to remove outliers.

        Parameters
        ----------
        self.outlier_perc : integer
            Percentile value for mstats.scoreatpercentile function. Mask all values greater than this value.
        """
        # Outliers using percentiles - num_rows * [min, max]
        outlier_all = ma.array([[mstats.scoreatpercentile(self.xs[i, :], 100 - self.outlier_perc),
               mstats.scoreatpercentile(self.xs[i, :], self.outlier_perc)] for i in xrange(self.rows_N)])
        self.xs = ma.array([ma.hstack((ma.masked_outside(self.xs[i, :-self.keep_n_values], outlier_all[i, 0],
            outlier_all[i, 1]), self.xs[i, -self.keep_n_values:])) for i in xrange(self.rows_N)])
    
    def clean_gaps_w_lin_regress(self, start_idx):
        """
        Function to clean gaps in the data with a linear regression.

        Parameters
        ----------
        start_idx : integer
            First non-masked value of array.
        """
        non_zero_idx = np.transpose(self.xs.nonzero())
        for i in xrange(self.rows_N):
            idx = non_zero_idx[np.where(non_zero_idx[:, 0] == i)][:, 1]
            if idx.any():
                slope, intercept, r, p, se = mstats.linregress(self.yrs[idx], self.xs[i,idx])
                missing_xs = ma.where(self.xs[i, start_idx[i]:-self.keep_n_values].mask)[0] + start_idx[i]
                if np.any(missing_xs):
                    self.xs[i, missing_xs] = (self.min_year + missing_xs) * slope + intercept
    
    def clean_gaps_w_linspace(self, fill_gap_length=0):
        """
        Function to fill gaps with linearly spaced values.

        Parameters
        ----------

        fill_gap_length : integer
            Maximum length of gaps to be filled by averaging.
        """
        self.xs = ma.masked_less_equal(self.xs, 0.)
        condition = ma.getmask(self.xs)

        # Fill masked value gaps if not at the beginning or end of the array or if the
        # gap length is less than the max gap length
        if np.any(condition):
            cont_regions = self.contiguous_regions(condition)
            for step in np.arange(2, np.size(cont_regions, 0) + 1, 2): # 1st row start, 2nd stop
                (axis, blank), (start, stop) = cont_regions[step - 2:step].T
                gap_length = stop - start # Length of gap
                if start and stop < self.N: # Don't fill in gaps at beginning or end
                    if not fill_gap_length or gap_length <= fill_gap_length:
                        self.xs[axis, start:stop] = np.linspace(self.xs[axis, start -1],
                            self.xs[axis, stop], stop - start + 1, endpoint=False)[1:]
    
    def find_new_starting_value(self):
        """
        Function to mask values at the beginning of the array if they have greater than
        max_masked_neighbors number of masked values after them.

        Parameters
        ----------
        self.masked_neighbors : integer
            Maximum number of masked values falling after a non-masked value.
        """
        mask = ~self.xs.mask
        shift_coeffs = [-x for x in xrange(1, self.masked_neighbors + 1)]
        for shift in shift_coeffs:
            x_shifted = np.roll(self.xs, shift=shift, axis=1)
            mask *= ~x_shifted.mask
        start_idx = ma.argmax(mask, axis=1) # Get first True or masked value for new starting year
        for i, s_idx in enumerate(start_idx):
            self.xs.mask[i, :s_idx] = True # Mask all values before starting index
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
            is_true_rows, = np.where(condition[:, 0]) # Row with mask at start
            for i in is_true_rows: # Append to end of the array
                idx = np.vstack((idx, [i, 0]))
        if np.any(condition[:, -1][:, np.newaxis]):
            is_true_rows, = np.where(condition[:, -1]) # Row with mask at end
            for i in is_true_rows: # Append to end of the array
                idx = np.vstack((idx, [i, np.size(condition, 1) - 1]))

        # Convert to structure array and then sort by indices, first by start and then stop
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
        return self.xs

class ExpSmooth:
    """
    Exponential trend smoothing class
    Example usage:
    from optimized_forecasting import ExpSmooth
    fCastPeriods = 20 # Number of periods to forecast
    f = ExpSmooth() # Initialize exponential smoothing class
    fitOpt = f.calc_variable_arrays(.98, xs, fCastPeriods) # Initialize arrays once and run optimization routine
    xs = f.exp_smooth_forecast(fitOpt, True) # Get exp smooth forecasted
    """
    def __init__(self):
        self.x0 = (0.25, 0.50, .98) # (trend_weight, levelWeight, trend_modifier) - starting values
        self.constr1 = lambda x0: .98 - x0[2]  # Constraint 1: trend_modifier < .98
        self.constr2 = lambda x0: x0[2] - .98  # Constraint 2: trend_modifier > .98
        self.constr3 = lambda x0: .95 - x0[1]  # Constraint 3: level_weight < .95
        self.constr4 = lambda x0: x0[1] - .05  # Constraint 4: level_weight > .05
        self.constr5 = lambda x0: .25 - x0[0]  # Constraint 5: trend_weight < .25
        self.constr6 = lambda x0: x0[0] - 0.   # Constraint 6: trend_weight > 0.
        self.constrs = [self.constr1, self.constr2, self.constr3,
                        self.constr4, self.constr5, self.constr6
        ]

    def calc_variable_arrays(self, trend_mod, data, periods):
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
        self.xs = data
        self.fcast_periods = periods
        self.data_N = self.xs.size
        self.warmUp = self.data_N / 2 # Going with a warm-up of half the data, instead of 6
        self.zero_len = self.data_N + self.fcast_periods # Length of arrays with one missing value
        self.lvlArr = np.zeros(self.zero_len + 1)
        self.errors = np.zeros(self.data_N)
        self.trendArr = np.zeros(self.zero_len + 1)
        self.fcasts = np.zeros(self.zero_len)
        avg_first_four_diffs = np.average(np.diff(self.xs[:4 + 1]))
        self.trendArr[0] = avg_first_four_diffs
        self.lvlArr[0] = self.xs[0] - avg_first_four_diffs
        self.trend_modifier = trend_mod
        self.trend_exp = np.ones(self.zero_len + 1)
        self.trend_exp[self.data_N:] = np.power(
            self.trend_modifier, (np.arange(self.data_N + 1, self.zero_len + 2) - self.data_N)
        )
        self.trend_multiplier = np.zeros(self.zero_len + 1)
        self.trend_multiplier[self.data_N:] = np.cumsum(self.trend_exp[self.data_N:])
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
        self.trend_weight, self.level_weight, self.trend_modifier = x0
        for i in xrange(1, self.data_N + 1):
            tmp = self.lvlArr[i - 1] + self.trendArr[i - 1] * self.trend_modifier
            self.fcasts[i - 1] = (not tmp < 0) * tmp
            self.errors[i - 1] = self.xs[i - 1] - self.fcasts[i - 1]
            self.trendArr[i] = self.trend_modifier * self.trendArr[i - 1] + self.trend_weight * self.errors[i - 1]
            self.lvlArr[i] = self.fcasts[i - 1] + self.level_weight * self.errors[i - 1]

        # Return forecast array concatenated with original data array, if final
        # else return forecasting MSE for the purpose of minimizing the function
        if final:
            self.lvlArr[self.data_N + 1:] = self.lvlArr[self.data_N]
            self.trendArr[self.data_N + 1:] = self.trendArr[self.data_N]
            self.fcasts[self.data_N - 1:] = self.lvlArr[self.data_N:] +\
                                            self.trendArr[self.data_N:] * self.trend_multiplier[self.data_N:]
            self.fcasts = np.ma.masked_less_equal(self.fcasts, 0)
            return np.ma.hstack((self.xs[:self.data_N], self.fcasts[self.data_N:]))
        else:
            sum_abs_errs_2 = np.cumsum(self.errors**2)
            warm_up_SSE = sum_abs_errs_2[self.warmUp - 1]
            fcast_SSE = sum_abs_errs_2[self.data_N - 1]
            SSE_diff = fcast_SSE - warm_up_SSE
            return SSE_diff / self.fcast_periods

class MiscForecastFunctions:
    """
    Miscellaneous forecasting functions:
    --> Polynomial trend - default is a one-degree polynomial aka a linear regression
    """
    def linear_regression(self, x, y, periods):
        """
        Function to forecast with a linear trend.

        Parameters
        ----------
        x : numpy masked array
            Vector of formatted year values
        y : numpy masked array
            Vector of formatted y values
        periods : integer
            Number of periods to forecast.
        """
        slope, intercept, _, _, _ = linregress(x, y)
        max_t = np.max(x) + 1 # Maximum time periods plus one for numpy range function
        forecast_periods = np.ma.arange(max_t, periods + max_t) # Periods to forecast
        forecasts = intercept + forecast_periods * slope
        return np.hstack((y, forecasts))

    def cons_forecast_multiple(self, consumption_xs, gdp_xs, elasticity):
        """
        Function to forecast consumption and per capita consumption values using elasticity and income.

        Parameters
        ----------
        consumption_xs : numpy masked array
            N-Dimensional array of formatted consumption values
        gdp_xs : numpy masked array
            N-Dimensional array of formatted per capita GDP (income) values.
        elasticity : numpy array
            Vector of elasticity values.
        """

        # Get the maximum non-masked value in the consumption array
#        np.transpose(np.ma.notmasked_edges(x, axis=1))[1:]
        edges = np.ma.notmasked_edges(consumption_xs, axis=1)
        gdp_xs = gdp.reshape(n, -1)
        cons_xs = cons.reshape(n, -1)
        elasticity_xs = elasticity.reshape(n, -1)
        fill = np.zeros((n, periods))
        d_ln = np.exp(elasticity_xs * np.diff(np.log(gdp_xs[:, last_val_idx:])))

        # Calculate the first value to start recursion
        fill[:, 0] = cons_xs[:, last_val_idx] * d_ln[:, 0]

        # Use the previously calculated consumption values to calculate future values
        for i in xrange(n):
            for j in xrange(1, periods):
                fill[i, j] = fill[i, j - 1] * d_ln[i, j]
        return np.hstack((cons_xs[:, :last_val_idx + 1], fill))

    def cons_forecast(self, consumption_xs, income_xs, elasticity, stop):
        """
        Function to forecast consumption and per capita consumption values using elasticity and income.

        Parameters
        ----------
        consumption_xs : numpy masked array
            N-Dimensional array of formatted consumption values
        income_xs : numpy array
            Vector of forecasted and formatted per capita GDP (income) values.
        elasticity : float
            Elasticity value.
        stop : integer
            Index of the last consumption value for the specified row.
        """

        # Number of periods to forecast
        periods = income_xs[stop + 1:].size

        # Empty array to hold forecasts
        fill_xs = np.zeros(periods)

        # Calculate the log differences of the income values that are in the forecasting range
        d_ln = np.exp(elasticity * np.diff(np.log(income_xs[stop:])))

        # Calculate the first value to start recursion
        fill_xs[0] = consumption_xs[stop] * d_ln[0]

        # Use the previously calculated consumption values to calculate future values
        for period in xrange(1, periods):
            fill_xs[period] = fill_xs[period - 1] * d_ln[period]
        return np.hstack((consumption_xs[:stop + 1], fill_xs))