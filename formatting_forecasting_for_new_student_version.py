"""
Formatting and forecasting script for new and final Python student version
"""
import numpy as np
from scipy import stats
from clean_data import CleanData, ExpSmooth
from sqlite3 import dbapi2 as sqlite

# Global variables
VALUE_COLUMN = 5 # First column that holds values
START_YEAR = 1961 # First year in the database
END_YEAR = 2030 # Last year in the database
NUM_FORECASTS = 7 # Number of forecast methods
ID_SLICE = slice(0, VALUE_COLUMN) # Slice for where id columns are located
X = np.arange(START_YEAR, END_YEAR) # Valid years for database
TABLE = "Datum" # Table to update in the database
exp_smooth = ExpSmooth() # Exponential smoothing class

# Connect to the database and create a cursor
DB = r"C:\Users\calanoue\Dropbox\Dont Want to Lose\GFIN Random Python Work\demoGFIN\sqlite_student_db.db3"
connection = sqlite.connect(DB)
cursor = connection.cursor()

# Get data from student database for formatting and forecasting - minus id column
Q = "SELECT * FROM Datum WHERE element_id NOT BETWEEN 511 AND 703"
datum_xs = np.ma.masked_equal(cursor.execute(Q).fetchall(), -1)[:, 1:]

def forecast_from_trend_line(xs, yrs, forecast_yrs, forecast_periods, trend_function):
    """
    Forecast data by using the specified trend function. Trend functions are the same functions offered in Excel
    for adding trend lines to a plot.
    """
    if trend_function == 1: # Linear trend (y = ax + B)
        slope, intercept, _, _, _ = stats.linregress(yrs, xs)
        y = slope * forecast_yrs + intercept
    elif trend_function == 2: # 2nd degree Polynomial trend (p(x) = p[0] * x**2 + p[2])
        z = np.polyfit(yrs, xs, 2)
        y = np.polyval(z, forecast_yrs)
    elif trend_function == 3: # 3rd degree Polynomial trend (p(x) = p[0] * x**3 + x**2 + p[3])
        z = np.polyfit(yrs, xs, 3)
        y = np.polyval(z, forecast_yrs)
    elif trend_function == 4: # Logarithmic trend (y = A + B log x)
        slope, intercept, _, _, _ = stats.linregress(np.log(yrs), xs)
        y = intercept + slope * np.log(forecast_yrs)
    elif trend_function == 5: # Exponential trend (y = Ae^(Bx))
        slope, intercept, _, _, _ = stats.linregress(yrs, np.log(xs))
        y = np.exp(intercept) * np.exp(slope * forecast_yrs)
    elif trend_function == 6: # Power function trend (y = Ax^B)
        slope, intercept, _, _, _ = stats.linregress(np.log(yrs), np.log(xs))
        y = np.exp(intercept) * np.power(forecast_yrs, slope)
    elif trend_function == 7: # Exponential smoothing with a dampened trend
        xs_fit_opt = exp_smooth.calc_variable_arrays(.98, xs, forecast_periods)
        y = exp_smooth.exp_smooth_forecast(xs_fit_opt, True)[-forecast_periods:]
    else: # Consumption forecasting with elasticity and income
        y = 8

    # Mask any negative, zero, infinity, or n/a values before returning
    y = np.ma.masked_less_equal(y, 0)
    y = np.ma.fix_invalid(y)
    return y

# Format all rows
new_datum_xs = np.ma.masked_all(datum_xs.shape, np.float)
count = 0
for row in datum_xs:
    try:
        start, stop = np.ma.flatnotmasked_edges(row[VALUE_COLUMN:][np.newaxis, :])
        values = CleanData(row[VALUE_COLUMN:stop + VALUE_COLUMN + 1][np.newaxis, :], X)
        xs = np.ma.hstack((values.get_return_values().flatten(), np.ma.masked_all(X.shape[0] - stop - 1)))
    except TypeError: # Some GDP rows do not have any values, therefore remove them
        xs = np.ma.array([0])
    if np.ma.sum(xs):
        new_datum_xs[count] = np.ma.hstack((row[ID_SLICE], xs))
        count += 1

# Resize the array to remove blank rows of data
new_datum_xs = np.ma.resize(new_datum_xs, (count, new_datum_xs.shape[1]))

# Append population and population net change arrays to the formatted and forecasted datum table
count = 0
Q = "SELECT * FROM Datum WHERE element_id BETWEEN 511 AND 703"
pop_xs = np.ma.masked_equal(cursor.execute(Q).fetchall(), -1)[:, 1:]
pop_xs = np.ma.filled(np.ma.column_stack(
    (np.ma.arange(
        count, pop_xs.shape[0]
    ), pop_xs[:, ID_SLICE], np.ma.masked_all((pop_xs.shape[0], 1)), pop_xs[:, VALUE_COLUMN:])
), -1)
count += pop_xs.shape[0]

# Add new column in the datum table for forecasting method values when adding the new trend data
Q = """
DROP TABLE Datum;
CREATE TABLE %s (id INTEGER PRIMARY KEY AUTOINCREMENT,country_id INTEGER REFERENCES Country,
item_id INTEGER REFERENCES Item,element_id INTEGER REFERENCES Element,unit_id INTEGER REFERENCES Unit,
source_id INTEGER REFERENCES Source,forecast_id INTEGER REFERENCES Forecast, yr1961 FLOAT,yr1962 FLOAT,yr1963 FLOAT,
yr1964 FLOAT,yr1965 FLOAT,yr1966 FLOAT,yr1967 FLOAT,yr1968 FLOAT,yr1969 FLOAT,yr1970 FLOAT,yr1971 FLOAT,yr1972 FLOAT,
yr1973 FLOAT,yr1974 FLOAT,yr1975 FLOAT,yr1976 FLOAT,yr1977 FLOAT,yr1978 FLOAT,yr1979 FLOAT,yr1980 FLOAT,yr1981 FLOAT,
yr1982 FLOAT,yr1983 FLOAT,yr1984 FLOAT,yr1985 FLOAT,yr1986 FLOAT,yr1987 FLOAT,yr1988 FLOAT,yr1989 FLOAT,yr1990 FLOAT,
yr1991 FLOAT,yr1992 FLOAT,yr1993 FLOAT,yr1994 FLOAT,yr1995 FLOAT,yr1996 FLOAT,yr1997 FLOAT,yr1998 FLOAT,yr1999 FLOAT,
yr2000 FLOAT,yr2001 FLOAT,yr2002 FLOAT,yr2003 FLOAT,yr2004 FLOAT,yr2005 FLOAT,yr2006 FLOAT,yr2007 FLOAT,yr2008 FLOAT,
yr2009 FLOAT,yr2010 FLOAT,yr2011 FLOAT,yr2012 FLOAT,yr2013 FLOAT,yr2014 FLOAT,yr2015 FLOAT,yr2016 FLOAT,yr2017 FLOAT,
yr2018 FLOAT,yr2019 FLOAT,yr2020 FLOAT,yr2021 FLOAT,yr2022 FLOAT,yr2023 FLOAT,yr2024 FLOAT,yr2025 FLOAT,yr2026 FLOAT,
yr2027 FLOAT,yr2028 FLOAT,yr2029 FLOAT,yr2030 FLOAT);
"""%TABLE
cursor.executescript(Q)

# Insert population data into Datum table
cursor.executemany(
    "INSERT INTO %s VALUES(%s)"%(TABLE, ','.join('?' for _ in xrange(pop_xs.shape[1]))), pop_xs
)
connection.commit()

# Extract the value and the id data from the returned query
values = new_datum_xs[:, VALUE_COLUMN:]
ids = new_datum_xs[:, ID_SLICE]

# Go through each row of remaining data - except population - and forecast using trend line methods above
# Add a new column at the end of ids to keep track of the forecasting trend method
N = new_datum_xs.shape[0]
for forecast_method in xrange(1, NUM_FORECASTS + 1):
    trend_datum_xs = np.ma.masked_all((N, new_datum_xs.shape[1] + 1), np.float)
    for enum, value_row in enumerate(values):
        xs = value_row[~value_row.mask]
        yrs = X[~value_row.mask]
        forecast_yrs = np.arange(np.max(yrs) + 1, END_YEAR + 1)
        forecast_periods = forecast_yrs.shape[0]

        # Forecast one method at a time
        trend_xs = forecast_from_trend_line(xs, yrs, forecast_yrs, forecast_periods, forecast_method)

        # Add masked values to the start if minimum starting year is greater than the first year
        trend_datum_xs[enum] = np.ma.hstack(
            (ids[enum], forecast_method, np.ma.masked_all(np.min(yrs) - START_YEAR), xs, trend_xs)
        )

    # Sort on index columns - forecast, element, item, country
    dtype = ",".join('<f8' for _ in xrange(trend_datum_xs.shape[1]))
    trend_datum_xs = np.ma.sort(trend_datum_xs.view(dtype), order=['f5', 'f2', 'f1', 'f0'], axis=0).view(np.float)

    # Add in a primary key field
    trend_datum_xs = np.ma.column_stack((np.ma.arange(count, count + N), trend_datum_xs))

    # Change missing values to -1 for storage in database
    trend_datum_xs = np.ma.filled(trend_datum_xs, -1)

    # Insert forecasted data into Datum table
    cursor.executemany(
        "INSERT INTO %s VALUES(%s)"%(TABLE, ','.join('?' for _ in xrange(trend_datum_xs.shape[1]))), trend_datum_xs
    )
    connection.commit()

    # Increase count of records for primary key index
    count += N

# TODO - change x and z coordinates to -1 for some of the countries or just find their centroids

# Add index to Datum table
cursor.execute("CREATE INDEX %s_index ON %s (forecast_id, element_id, item_id, country_id)"%(TABLE, TABLE))
connection.commit()

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

# Close the cursor and the connection
cursor.close()
connection.close()
