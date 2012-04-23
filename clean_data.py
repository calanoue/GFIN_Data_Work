"""
Format and clean data.
"""
import numpy as np
import numpy.ma as ma
from scipy.stats import linregress, mstats
from itertools import imap

class CleanData:
    """
    Class to hold functions corresponding to cleaning and formatting of data.
    """
    def __init__(self, data, years, recents=7, perc_remove_1=.8, gap_length_to_fill=4,
                 outlier_p=98, outlier_t=False, max_neighbors=3, perc_remove_2=.6,
                 perc_remove_3=.65):
        self.xs = data # array of data
        self.yrs = years # years corresponding to the data
        self.num_rows = np.size(self.xs, 0) # number of rows
        self.N = np.size(self.xs, 1) # length of the data array
        self.keep_n_values = recents # number of values at the end of the array to always keep
        self.xs_formats = np.zeros((self.num_rows, self.N), object) # array for holding formatting codes
        self.min_year = np.min(self.yrs) # minimum year
        self.init_perc_remove = perc_remove_1 # remove row if more masked values than this
        self.second_perc_remove = perc_remove_2 # remove row after outlier and average if greater than this
        self.perc_remove_after_start_idx = perc_remove_3 # remove row if after start index this many are masked
        self.max_gap_length = gap_length_to_fill # max gap length to fill with linspace
        self.outlier_perc = outlier_p # percentage for outlier scoreatpercentile calculations
        self.outlier_normal = outlier_t # calculate outlier by percentile or normal dist
        self.masked_neighbors = max_neighbors # max number of masked neighbors for new start
        self.format_and_clean_data_main() # run main function to format
    #-----------------------------------------------------------------------
    def format_and_clean_data_main(self):
        """
        Main function to format and clean data based on choices by the user.
        """
        # Check if over missing_bound percent or missing_bound number of values are missing
        too_many_missing = self.has_too_many_missing(self.init_perc_remove) # boolean
        if ma.any(too_many_missing):
            idx, = ma.where(too_many_missing)
            self.xs[idx] = ma.mask_rows(self.xs[idx])

        # Check array to see if it doesn't need to be formatted or cleaned, i.e. either all values
        # are masked or all are non-masked
        if ma.all(self.check_for_all()):
            return self.xs, self.xs_formats

        # Clean outliers
        self.clean_outliers() # code ==> 1

        # Take average of neighbor values to fill up to a given missing value gap length
        self.clean_gaps_w_linspace(fill_gap_length=self.max_gap_length) #code ==> 2
        if ma.all(ma.count_masked(self.xs[:,:-self.keep_n_values], axis=1)[np.newaxis,:]==0):
            return self.xs, self.xs_formats # if no masked values remain in values before recent ones

        # Remove values if they start the array and are then followed by too many masked values
        start_idx = self.find_new_starting_value() # code ==> 3

        # If there are over x% blank values left in the original data after above changes,
        # check to see if x% of the blanks fall after the new start year
        too_many_missing = self.has_too_many_missing(self.second_perc_remove) # boolean array
        if ma.any(too_many_missing):
            n_masked = np.array([ma.count_masked(self.xs[i,s_idx:])
                                 for i, s_idx in enumerate(start_idx)])/self.N > self.perc_remove_after_start_idx
            if ma.any(n_masked):
                idx, = ma.where(n_masked)
                self.xs[idx] = ma.mask_rows(self.xs[idx])
                self.xs_formats[idx] = np.zeros(self.N)

        # To fill in remaining values, run linear regression on non-zero values
        self.clean_gaps_w_lin_regress(start_idx) # code ==> 4

        # If linear regression left negative or zero values,
        # then use linspace to fill in middle gaps
        if ma.any(ma.masked_less_equal(self.xs, 0.)):
            self.clean_gaps_w_linspace() # code ==> 5
    #-----------------------------------------------------------------------
    def check_for_all(self):
        """
        Function to check if all the values are masked or all not masked
        """
        masked_N = ma.count_masked(self.xs, axis=1)[np.newaxis,:]
        is_all_masked = self.N==masked_N
        is_none_masked = masked_N==0
        return is_none_masked + is_all_masked
    #-----------------------------------------------------------------------
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
    #-----------------------------------------------------------------------
    def clean_outliers(self):
        """
        Function to remove outliers.

        Parameters
        ----------
        self.outlier_perc : integer
            Percentile value for scipy.scoreatpercentile function. Mask all values greater
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
            std = self.outlier_std_dev*self.xs.std(1)
            outlier_all = np.reshape((self.xs.mean(1) + [-std, std]).T, (self.num_rows, 2))
        else: # outliers using percentiles - num_rows * [min, max]
            outlier_all = ma.array([[mstats.scoreatpercentile(self.xs[i,:], 100 - self.outlier_perc),
                   mstats.scoreatpercentile(self.xs[i,:], self.outlier_perc)] for i in xrange(self.num_rows)])
        self.xs = ma.array([ma.hstack((ma.masked_outside(self.xs[i,:-self.keep_n_values], outlier_all[i,0],
            outlier_all[i,1]), self.xs[i,-self.keep_n_values:])) for i in xrange(self.num_rows)])
        idx = ~hldr.mask*self.xs.mask # boolean array of changes
        if ma.any(idx): # if there have been any outlier removals
            self.fill_xs_formats(idx, 1)
    #-----------------------------------------------------------------------
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
        for i in xrange(self.num_rows):
            idx = non_zero_idx[np.where(non_zero_idx[:,0]==i)][:,1]
            if idx.any():
                slope, intercept, r, p, se = linregress(self.yrs[idx], self.xs[i,idx])
                missing = ma.where(self.xs[i, start_idx[i]:-self.keep_n_values].mask)[0] + start_idx[i] # missing vals
                if np.any(missing):
                    self.xs[i, missing] = (self.min_year + missing)*slope + intercept
        idx = hldr.mask*~self.xs.mask
        if ma.any(idx): # if any values are filled with the linear regression
            self.fill_xs_formats(idx, 4)
    #-----------------------------------------------------------------------
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
        idx = hldr.mask*~self.xs.mask
        if ma.any(idx): # if any values are filled with linear spacing
            self.fill_xs_formats(idx, 5)
    #-----------------------------------------------------------------------
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
            self.xs.mask[i,:s_idx] = True # mask all values before starting index
        idx = ~hldr.mask*self.xs.mask
        if ma.any(idx): # if eliminated any values due to falling before new starting year
            self.fill_xs_formats(idx, 3)
        return start_idx
    #-----------------------------------------------------------------------
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
        idx = np.transpose(diff.nonzero()) # Find the indicies of changes in condition
        idx[:, 1] += 1 # Shift the column index by 1 to the right.
        if np.any(condition[:,0][:,np.newaxis]):
            is_true_rows, = np.where(condition[:,0]) # row with mask at start
            for i in is_true_rows: # append to end of the array
                idx = np.vstack((idx, [i, 0]))
        if np.any(condition[:, -1][:,np.newaxis]):
            is_true_rows, = np.where(condition[:, -1]) # row with mask at end
            for i in is_true_rows: # append to end of the array
                idx = np.vstack((idx, [i, np.size(condition, 1) - 1]))

        # Convert to recarray and then sort by indices, first by start and then stop
        idx = np.array([i for i in imap(tuple, idx)], dtype=[('start',int), ('stop',int)])
        idx = np.sort(idx, order=['start', 'stop']).view(int).reshape((-1, 2))
        return idx
    #-----------------------------------------------------------------------
    def has_too_many_missing(self, missing_bound):
        """
        Function to find if over missing_bound percent or
        missing_bound number of values are missing

        Parameters
        ----------
        missing_bound : if < 1 ==> float, otherwise integer
            Bound for checking if too many values are missing in the array.
        """
        masked_count = ma.count_masked(self.xs, axis=1)*1.
        return [masked_count > missing_bound,
                masked_count/self.N > missing_bound][missing_bound < 1]
    #-----------------------------------------------------------------------
    def get_return_values(self):
        """
        Function to return the values and formats array.
        """
        return self.xs, self.xs_formats
#-----------------------------------------------------------------------
# Get data from database for formatting and then forecast
#-----------------------------------------------------------------------
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
    start, stop = k[1,:]
    periods = forecast_year - last_year + (N - stop)
    xs = x[i, start:stop + 1]
    fitOpt = f.calc_variable_arrays(.98, xs, periods) # initialize arrays once and run optimization routine
    xs = ma.hstack((ma.masked_all(start), f.exp_smooth_forecast(fitOpt, True)))

# Close cursor and connection
cursor.close()
connection.close()