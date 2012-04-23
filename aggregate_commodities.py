"""
Aggregate commodities based on their FAO aggregation relationships.
"""
import numpy as np
import sqlite3
from itertools import izip

# Connect to the database and create a cursor for extracting data
DB = r".\GFIN_DB.db3"
TABLE_NAME = "Commodity"
connection = sqlite3.connect(DB)
cursor = connection.cursor()

# Get all item codes that need to be aggregated into another commodity
ys = np.genfromtxt(".\Commodity Code Conversions\CCode_to_ProdStat.csv", delimiter=",", usemask=True, dtype=None)
unique_aggregate_items = np.unique(ys) # items that need to be summed, averaged, etc
aggregate_item_lookup = dict(izip(ys[:,0], ys[:,1:])) # dictionary of item codes, {main_item_id:aggregate_item_ids}

# Get items that don't need to be aggregated and run the query at the end to add to aggregated array
unique_items = np.array(cursor.execute("SELECT item_id FROM Item").fetchall()).flatten()
keep_items_SQL = "SELECT %%s FROM %s WHERE %s"%(TABLE_NAME, " OR ".join("item_id=%s"%item for item in
    [item for item in unique_items if item not in unique_aggregate_items]))

# Data types and names for the final array with all data
names = ["country_id", "item_id", "element_id", "unit_id", "source_id"] + ["yr%s"%x for x in np.arange(1961, 2011)]
dtype = zip(names, ['<i4']*len(names)) # data type for all data in database
all_xs = np.empty((330510,), dtype=dtype) # array to hold all aggregated data
init_SQL = "SELECT %s FROM %s WHERE %%s"%(",".join(names), TABLE_NAME) # initial SQL query

# Go through each key and value array in the aggregate dictionary and sum/average the arrays
count = 0
for key, values in aggregate_item_lookup.iteritems():
    key_tracker = [] # keep track of what key, element, country ids should be removed
    N = np.transpose(np.ma.notmasked_edges(values, axis=1))[1] + 1 # number of items to aggregate
    agg_items_SQL = " OR ".join("item_id=%s"%v for v in values[:N])
    values_xs = np.array(cursor.execute(init_SQL%agg_items_SQL).fetchall(), dtype=dtype)
    values_xs['item_id'] = key
    if N==1: # insert back into main data array
        values_xs_n = np.size(values_xs, 0) # number of rows in the array
        all_xs[count:count + values_xs_n] = values_xs # insert changed rows into main data array
        count += values_xs_n
    else: # aggregate the items up to the key level
        elements = np.unique(values_xs['element_id'])
        for element in elements:
            f = [np.ma.sum, np.ma.average][element in (41, 60)] # aggregation function for yield and price
            element_xs = values_xs[np.where(values_xs['element_id']==element)] # smaller data-set under each element
            countries = np.unique(element_xs['country_id']) # unique countries for each element
            for country in countries:
                country_xs = np.ma.masked_less_equal(
                    element_xs[np.where(element_xs['country_id']==country)].view(int).reshape(-1,
                        len(names)), 0)
                country_agg_xs = tuple(np.ma.filled(np.ma.hstack((country_xs[0,:5],
                                                                     f(country_xs[:,5:], axis=0))), -1).flatten())
                all_xs[count:count + 1] = country_agg_xs
                count += 1 # aggregated down to one record
                if element==51: # if it is production and a country/element row has been updated
                    key_tracker.append(country)

    # If production elements were aggregated above, remove corresponding FBS production row
    key_item_SQL = "item_id=%s"%key
    if key_tracker:
        key_item_SQL += " AND not (%s)"%" OR ".join("(country_id=%s AND element_id=51)"%country for country in key_tracker)

    # Append remaining production and consumption FBS data
    key_xs = np.array(cursor.execute(init_SQL%key_item_SQL).fetchall(), dtype=dtype)
    key_xs_n = np.size(key_xs, 0)
    all_xs[count:count + key_xs_n] = key_xs
    count += key_xs_n

# Insert data that doesn't need to be aggregated into the main data array
keep_xs = np.array(cursor.execute(keep_items_SQL%",".join(names)).fetchall(), dtype=dtype)
keep_xs_n = np.size(keep_xs, 0)
all_xs[count:count + keep_xs_n] = keep_xs
count += keep_xs_n # final count of all values

# Resize final array to eliminate empty rows
all_xs = np.resize(all_xs, (count,))

# Sort according to the order of the indexes
all_xs = np.sort(all_xs, order=['source_id', 'element_id', 'item_id', 'country_id'])

# Drop Commodity Table to add in updated table using sqlite_io
cursor.execute("DROP TABLE %s"%TABLE_NAME)
connection.commit()

# Close cursor and connection
cursor.close()
connection.close()

# Create new Commodity table for aggregate data
import sqlite_io

foreign_keys = {
    'country_id':'Country', 'item_id':'Item', 'element_id':'Element',
    'unit_id':'Unit', 'source_id':'Source'
}
index = ['source_id', 'element_id', 'item_id', 'country_id'] # index in order
sqlite_io.tosqlite(all_xs, 0, DB, TABLE_NAME, autoid=True, foreign_keys=foreign_keys, index=index, create=True)