"""
Calculate per capita consumption and production values for the Commodity Table.
"""
import numpy as np
from sqlite3 import dbapi2 as sqlite
from get_numpy_dtype import get_dtype
from itertools import imap

# Globals
COUNTRY_YR_IDX = np.arange(1961, 2011)

# Connect to the database and create a cursor
DB = r".\GFIN_DB.db3"
connection = sqlite.connect(DB)
connection.text_factory = str #use 8 bit strings instead of unicode strings in SQLite
cursor = connection.cursor()

# Get all Consumption and Production Data and mask values less than or equal to 0
ndtype, names = get_dtype(connection, "Commodity", nameReturn=True, remove_id=True)
#TODO - figure out what to do with region values
Q = "SELECT %s FROM Commodity WHERE (element_id=51 OR element_id=100) AND country_id < 5000"%",".join(names)
commodity_xs = np.ma.array(cursor.execute(Q).fetchall(), ndtype)
unique_country_ids = np.unique(commodity_xs['country_id']) # unique country ids

# Array to hold all values to insert into Commodity table
insert_xs = np.ma.empty((100000, len(names)))

# Demographic table names and data types
demo_names = ",".join("yr%s"%x for x in COUNTRY_YR_IDX) # yr1961, yr1962, ..., yr2009, yr2010
demo_ndtype = zip(demo_names.split(","), len(COUNTRY_YR_IDX)*['<f8'])

# Go through each country in the Commodity table
count = 0
for country_id in unique_country_ids:

    # Get Total Population - Both Sexes (511) data and mask values less than or equal to 0 by country
    Q = "SELECT %s FROM Demographic WHERE element_id=511 AND country_id=%s"%(demo_names, country_id)
    pop_xs = np.ma.masked_less_equal(
        np.ma.array(cursor.execute(Q).fetchall(), demo_ndtype).view(float).reshape(-1, len(COUNTRY_YR_IDX)),
    0).flatten()

    # Get commodity data by country
    xs = np.ma.masked_less_equal(commodity_xs[np.where(commodity_xs['country_id']==country_id)].view(int).reshape(-1,
        len(names)), 0).astype(float)

    # Calculate per capita values and stack id values at the front
    tmp_xs = np.array(np.ma.filled(np.ma.hstack((xs[:,:5], np.ma.divide(xs[:,5:], pop_xs))), -1))

    # Insert values into Commodity Table and update count
    N = np.size(tmp_xs, 0)
    insert_xs[count:count + N] = tmp_xs
    count += N

# Resize insert array
insert_xs = np.resize(insert_xs, (count, len(names)))
print insert_xs, np.shape(insert_xs)

# Close the cursor and the connection
cursor.close()
connection.close()
