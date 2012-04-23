"""
Manipulate trade matrix data from FAO. Make sure columns in Excel have the proper number formatting.
Columns for database:
-----------
country_id, element_id, item_id, trade_country_id, source_id, yr1986, yr1987, ..., yr2008, yr2009
"""
import numpy as np
from numpy.lib.recfunctions import merge_arrays
from itertools import izip

# Global variables
DIR = ".\TradeSTAT - Trade Matrix\demo_TradeMatrix"
FILE_NAMES = ["%s_Exports.csv"%DIR, "%s_Imports.csv"%DIR] # import and export trade files
MIN_YEAR, MAX_YEAR = 1986, 2009 # min and max years of the trade data
YEARS = np.arange(MIN_YEAR, MAX_YEAR + 1) # all possible years for trades
SOURCE_ID = 7 # foreign key relating to this specific source
find_key = lambda dic, val: [k for k, v in dic.iteritems() if np.any(v==val, axis=0)]

# Get all item codes that need to be aggregated into another commodity
ys = np.genfromtxt(".\Commodity Code Conversions\CCode_to_ProdStat.csv", delimiter=",", usemask=True, dtype=None)
aggregate_item_lookup = dict(izip(ys[:,0], ys[:,1:])) # dictionary of item codes, {main_item_id:aggregate_item_ids}

# Initialize arrays to hold ids and trade values and quantities
num_rows = 256*12*8 # safe number of rows for initial array
trade_rows = np.ma.empty((num_rows),
    dtype=zip(["yr%s"%i for i in YEARS], ['<f8']*YEARS.size))
id_rows = np.ma.empty((num_rows), dtype=zip(["country_id", "element_id", "item_id", "trade_country_id", "source_id"],
        ["<i4", "<i4", "<i4", "<i4", "<i4"]))
count = 0 # start count at 0 for resizing purposes and because array index starts at 0

# Go through each file
for FILE_NAME in FILE_NAMES:
    # Get import and export data from the CSV files
    xs = np.genfromtxt(FILE_NAME, delimiter=",", names=True, usemask=True, dtype=None)

    # Go through all countries and their nested commodities to find all the trade relationships
    unique_country_ids = np.unique(xs['reporting_country_id'])
    column_names = xs.dtype.names[3:]
    for country_id in unique_country_ids:
        x = xs[np.where(xs['reporting_country_id']==country_id)] # all values for a reporting country
        unique_item_ids = np.unique(x['item_id']) # unique items by reporting country
        for item_id in unique_item_ids:
            y = x[np.where(x['item_id']==item_id)] # all values for a specified commodity
            for column in column_names:
                z = y[column].view(float) # all values in a column
                not_empty = np.transpose(np.ma.nonzero(z)) # does the column have any unmasked values?
                if np.any(not_empty): # if the country has a trade relationship
                    trade_row = np.ma.masked_all(YEARS.size, dtype=float) # masked array with all year values
                    trade_country_id, element_id = column.split("_") # split column on _ separator
                    idx = y['year_id']  - MIN_YEAR # index value of years to include
                    item_lookup = find_key(aggregate_item_lookup, item_id)[0] # lookup new item code
                    trade_row[idx] = z # match up and assign values based on year values
                    trade_rows[count] = trade_row
                    id_rows[count] = np.array([country_id, element_id, item_lookup, trade_country_id, SOURCE_ID],
                        dtype=int)
                    count += 1

# Resize arrays to match actual size of the data
trade_rows = np.ma.filled(np.ma.resize(trade_rows, (count,)), -1) # fill masked values with -1 for database
id_rows = np.ma.resize(id_rows, (count,))

# Merge and flatten the two arrays to keep the proper data types
xs_rows = merge_arrays((id_rows, trade_rows), flatten=True) # values for the database
xs_rows = np.sort(xs_rows, order=["element_id", "item_id", "country_id", "trade_country_id"])

# Create new database table for Trade Relationships
import sqlite_io

DB = r".\GFIN_DB.db3"
TABLE_NAME = "Trade"
foreign_keys = {
    'country_id':'Country', 'element_id':'Element','unit_id':'Unit',
    'source_id':'Source', 'trade_country_id':'Country'
}
index = ['element_id', 'item_id', 'country_id'] # index in order
sqlite_io.tosqlite(xs_rows, 0, DB, TABLE_NAME, autoid=True,
    foreign_keys=foreign_keys, index=index, create=True) # create a new table in the database for trade matrix
