"""
Create the GFIN_DB.db3 database by removing any row of the main database that doesn't have more
than 5 values. This is done by masking any values less than or equal to 0. Also, change the
masked values in the new database from <null> to -1 for optimizing queries.
"""
import numpy as np
from sqlite3 import dbapi2 as sqlite
import sqlite_io
from get_numpy_dtype import get_dtype, mask_none_values

# Global variables
DB = "./FBS_ProdSTAT_PriceSTAT_TradeSTAT.db3" # main database
NEW_DB = "./GFIN_DB.db3" # new database that will be created
TABLES = ["Commodity", "Demographic"] # tables to perform removal and re-formatting of masked values
REMOVE_IF_LESS_THAN = 5 # remove rows with values less than this

# Database connection and cursor
connection = sqlite.connect(DB)
connection.text_factory = str #use 8 bit strings instead of unicode strings in SQLite
cursor = connection.cursor()

# Copy all Foreign Key and Variable tables over to new database
copy_tables =  np.array(cursor.execute("""
SELECT name FROM sqlite_master
WHERE type='table' AND name!='sqlite_sequence' AND name!='Commodity' AND name!='Demographic'""").fetchall()
).flatten()
for table in copy_tables:
    is_autoid = table in ('SchemeColor', 'AreaGroup') # tables with id as primary key
    ndtype, names = get_dtype(connection, table, remove_id=is_autoid, nameReturn=True)

    # Get data from master database for copying
    xs = np.ma.array(cursor.execute("SELECT %s FROM %s"%(",".join(names), table)).fetchall(), ndtype)

    # Mask all None values and create primary keys
    autoid = [False, True][is_autoid] # assign primary keys
    index = [["%s_id"%table.lower()], []][is_autoid]
    primary_key = ["%s_id"%table.lower(), False][is_autoid] # primary key
    xs = mask_none_values(xs) # mask none values
    sqlite_io.tosqlite(xs, 0, NEW_DB, table, autoid=autoid,
        create=True, primary_key=primary_key, index=index)

# Format value tables with -1 values for missing values
for table in TABLES:
    (names, typestr) = zip(*(_[1:3] for _ in connection.execute("PRAGMA TABLE_INFO(%s)"%table).fetchall()))
    names = ",".join([name.strip() for name in names if name.strip()!='id'])
    xs = sqlite_io.fromsqlite(DB, "SELECT %s FROM %s"%(names, table), "tmp_table")
    ndtype = xs.dtype
    index = [
        ["source_id", "element_id", "item_id", "country_id"],
        ["source_id", "element_id", "country_id"]
    ][table=="Demographic"]
    foreign_keys = {i:i[:-3].capitalize() for i in index}
    xs =  xs.view(float).reshape((-1, len(names.split(","))))
    xs = np.ma.masked_less_equal(xs, 0) # mask any value less than or equal to 0

    # Remove Commodity rows that have less than 5 values
    if table=='Commodity':
        id_field_idx = 5 # number of columns that split the data b/w foreign keys and values
        id_fields = xs[:,:id_field_idx] # foreign key fields
        value_fields = xs[:,id_field_idx:] # data value fields
        keep_rows, = np.where((np.size(value_fields, 1) - np.ma.count_masked(value_fields, axis=1)) > REMOVE_IF_LESS_THAN)
        xs = np.ma.hstack((id_fields[keep_rows,:], value_fields[keep_rows,:]))

    # Fill masked values with a -1 value
    xs = np.ma.filled(xs, -1)

    # Convert ndarray to a recarray using view
    xs = xs.view(ndtype).flatten()

    # Insert values into the specified table
    sqlite_io.tosqlite(xs, 0, NEW_DB, table, autoid=True, create=True,
        foreign_keys=foreign_keys, index=index)

# Close the cursor and the connection
cursor.close()
connection.close()

