"""
Calculate per capita consumption and production values for the Commodity Table for both areas and countries.
"""
import numpy as np
from sqlite3 import dbapi2 as sqlite
from get_numpy_dtype import get_dtype
import sqlite_io

# Globals
COUNTRY_YR_IDX = np.arange(1961, 2011)

# Connect to the database and create a cursor
DB = r".\GFIN_DB.db3"
connection = sqlite.connect(DB)
connection.text_factory = str # use 8 bit strings instead of unicode strings in SQLite
cursor = connection.cursor()

# Get all Consumption and Production Data and mask values less than or equal to 0
ndtype, names = get_dtype(connection, "Commodity", nameReturn=True, remove_id=True)
Q = "SELECT %s FROM Commodity WHERE (element_id=51 OR element_id=100)"%",".join(names)
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
    xs = np.ma.masked_less_equal(
        commodity_xs[np.where(commodity_xs['country_id']==country_id)].view(float).reshape(-1, len(names)),
        0)

    # Calculate per capita values and stack id values at the front
    tmp_xs = np.ma.hstack((xs[:,:5], np.ma.divide(xs[:,5:], pop_xs)))

    # Insert values into Commodity Table and update count
    N = np.size(tmp_xs, 0)
    insert_xs[count:count + N] = tmp_xs
    count += N

# Resize insert array
insert_xs = np.resize(insert_xs, (count, len(names)))

# Fill masked values with -1
insert_xs = np.ma.filled(insert_xs, -1)

# Convert ndarray to recarray
insert_xs = insert_xs.view(ndtype).flatten()

# Replace element_ids with those for the per capita elements
np.put(insert_xs['element_id'], np.where(insert_xs['element_id']==100), 101) # per capita consumption
np.put(insert_xs['element_id'], np.where(insert_xs['element_id']==51), 52) # per capita production

# Replace unit_ids with those for the per capita elements
np.put(insert_xs['unit_id'], np.where(insert_xs['unit_id']==3), 17) # tonnes/1000 people
np.put(insert_xs['unit_id'], np.where(insert_xs['unit_id']==9), 18) # 1000 No/1000 people

# Replace source_ids with a new code for 'InnovoSoy Calculated'
insert_xs['source_id'] = 8

# Get last index value + 1 of Commodity table for primary key values
max_id, = np.array(cursor.execute("SELECT MAX(id) FROM Commodity").fetchall()).flatten() + 1

# Insert new data into the database using sqlite_io structure
sqlite_io.tosqlite(insert_xs, max_id, DB, "Commodity", autoid=True, create=False)

# Close the cursor and the connection
cursor.close()
connection.close()
