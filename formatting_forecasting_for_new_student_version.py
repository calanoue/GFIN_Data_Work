"""
Formatting and forecasting script for new and final Python student version
"""
import numpy as np
from clean_data import CleanData, ExpSmooth, MiscForecastFunctions
from sqlite3 import dbapi2 as sqlite

# Global variables
VALUE_SLICE = slice(5, None, 1)
ID_SLICE = slice(0, 5)
X = np.arange(1961, 2031)

# Connect to the database and create a cursor
DB = r"C:\Users\calanoue\Dropbox\Dont Want to Lose\GFIN Random Python Work\demoGFIN\sqlite_student_db.db3"
connection = sqlite.connect(DB)
connection.text_factory = str # use 8 bit strings instead of unicode strings in SQLite
cursor = connection.cursor()

# Items to keep in the new database
# TODO - create new database with only commodities from Peter
Q = "SELECT * FROM Datum WHERE (item_id=2555 OR item_id=-1) AND (element_id < 511 OR element_id > 703)"
datum_xs = np.ma.masked_equal(cursor.execute(Q).fetchall(), -1)[:, 1:]

# Countries to keep commodity, GDP, and Population data
# TODO - change x and z coordinates to -1 for some of the countries or just find their centroids

# Format all rows
new_datum_xs = np.ma.masked_all(datum_xs.shape, np.float)
count = 0
for row in datum_xs:
    values = CleanData(row[VALUE_SLICE][np.newaxis, :], X)
    xs = values.get_return_values().flatten()
    if np.ma.sum(xs):
        new_datum_xs[count] = np.ma.hstack((row[ID_SLICE], xs))
        count += 1

# Remove blank rows
new_datum_xs = np.ma.resize(new_datum_xs, (count, new_datum_xs.shape[1]))

# TODO - forecast here
# For non-consumption data:
# 1. Exp smoothing
# 2. Linear regression
# 3. 3 period MA
# 4. 3 period MA w/ weights - .2, .3, and .5 weights

# For consumption data:
# 1. Elasticity and income
# 2. Linear regression
# 3. Exp smoothing
# 4. 3 period MA
# 5. 3 period MA w/ weights - .2, .3, and .5 weights

# Append population and population net change arrays to the formatted and forecasted datum table
Q = "SELECT * FROM Datum WHERE element_id BETWEEN 511 AND 703"
pop_xs = np.ma.masked_equal(cursor.execute(Q).fetchall(), -1)[:, 1:]
new_datum_xs = np.ma.vstack((new_datum_xs, pop_xs))

# Sort on index columns
dtype = ",".join('<f8' for _ in xrange(new_datum_xs.shape[1]))
new_datum_xs = np.ma.sort(new_datum_xs.view(dtype), order=['f2', 'f1', 'f0'], axis=0).view(np.float)

# Add in a primary key field
new_datum_xs = np.ma.column_stack((np.ma.arange(new_datum_xs.shape[0]), new_datum_xs))
print new_datum_xs
exit()

def forecast_xs_consumption(formatted_xs, id_xs, country_id_index):
    """
    Forecast consumption values from income and elasticity.
    """
    xs = formatted_xs

    if not np.any(self.income_xs):

        # Get unique country ids from consumption values for income
        country_ids = np.unique(id_xs[:, country_id_index])

        # Retrieve and format income data
        self.income_xs = np.ma.masked_equal(
            self.cursor[self.c_cycle.next()].execute(self.sql.left_query_income(country_ids)).fetchall(), -1
        )
        income_x, income_x_formats = self.format_xs(self.income_xs[:, 6:])

        # Forecast income data using a linear regression
        # TODO - change to exp smooth forecast here
        income_edges = np.transpose(np.ma.notmasked_edges(income_x, axis=1))
        for i, row in enumerate(income_x):
            start, stop = income_edges[i, 1, :]
            slope, intercept, r_value, p_value, std_err = stats.linregress(X[:stop + 1], row[:stop + 1])
            forecasts = intercept + X[stop + 1:] * slope
            self.income_xs[i, 6:] = np.hstack((row[:stop + 1], forecasts))

    # Forecast consumption values using forecasted income and elasticity values
    consumption_edges = np.transpose(np.ma.notmasked_edges(xs, axis=1))
    for i, idx in enumerate(consumption_edges[:, 0, 0]):
        start, stop = consumption_edges[i, 1, :]
        country_id = id_xs[idx, country_id_index]

        # Attempt to find the income that goes with the specified and if not found return the original masked xs
        try:
            income_row = self.income_xs[np.where(self.income_xs[:, 1] == country_id)[0]].flatten()[6:]
            xs[idx] = self.misc_forecast.cons_forecast(xs[idx], income_row, TMP_ELASTICITY, stop)
        except IndexError:
            pass
    return xs

def forecast_xs_non_consumption(formatted_xs):
    """
    Forecast non-consumption values.
    """
    n = np.size(formatted_xs, 0)
    xs = np.ma.empty((n, X_N))
    edges = np.transpose(np.ma.notmasked_edges(formatted_xs, axis=1))
    for i, edge in enumerate(edges):
        start, stop = edge[1, :]
        periods = X_N - stop - 1
        if periods and np.ma.sum(formatted_xs[i, start:stop + 1]):
            xs_forecast = formatted_xs[i, start:stop + 1]
            xs_fit_opt = self.exp_smooth.calc_variable_arrays(.98, xs_forecast, periods)
            xs[i] = np.ma.hstack(
                (np.ma.masked_all(start), self.exp_smooth.exp_smooth_forecast(xs_fit_opt, True))
            )
        else:
            xs[i] = formatted_xs[i]
    return xs

# Close the cursor and the connection
cursor.close()
connection.close()
