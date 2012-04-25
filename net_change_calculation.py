"""
Calculate net change values for all population fields
"""
import numpy as np
from sqlite3 import dbapi2 as sqlite
from get_numpy_dtype import get_dtype

# Connect to the database and create a cursor
DB = r".\GFIN_DB.db3"
connection = sqlite.connect(DB)
connection.text_factory = str # use 8 bit strings instead of unicode strings in SQLite
cursor = connection.cursor()

# Get all demographic data fields as a masked array except for id field
select = """
SELECT * FROM Demographic WHERE element_id BETWEEN 511 AND 603
"""
xs = np.ma.masked_less_equal(np.array(cursor.execute(select).fetchall()), 0)[:,1:]

# Calculate masked differences along the first axis
diff_xs = np.ma.masked_all(np.shape(xs[:, 4:]))
diff_xs[:, 1:] = np.ma.diff(xs[:, 4:], axis=1)
diff_xs = np.ma.filled(diff_xs, -1)

# Stack id fields with net change values
xs = np.ma.filled(xs, -1)
ys = np.hstack((xs[:, :4], diff_xs))

# Convert ndarray to recarray
ys = ys.reshape(-1, ).view(get_dtype(connection, 'Demographic', remove_id=True))

# Replace element values w/ their corresponding net change values
for element_id in [511, 512, 513, 551, 561, 571, 581, 591, 592, 593, 601, 602, 603]:
    np.put(ys['element_id'], np.where(ys['element_id']==element_id), element_id + 100)

# Get last index value + 1 of Demographic table for primary key values
max_id, = np.array(cursor.execute("SELECT MAX(id) FROM Demographic").fetchall()).flatten() + 1

# Insert new data into the database using sqlite_io structure
import sqlite_io
sqlite_io.tosqlite(ys, max_id, DB, "Demographic", autoid=True, create=False)

# Close the cursor and the connection
cursor.close()
connection.close()