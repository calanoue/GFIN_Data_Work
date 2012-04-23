"""
Delete commodities from Item table that no longer exist in the Commodity table
"""

from numpy import array, setdiff1d
import sqlite3

# Connect to the database and create a cursor for extracting data
DB = r".\GFIN_DB.db3"
connection = sqlite3.connect(DB)
cursor = connection.cursor()

# Find which items need to be removed
keep_items = array(cursor.execute("SELECT DISTINCT item_id FROM Commodity").fetchall(), int).flatten()
all_items = array(cursor.execute("SELECT item_id FROM Item").fetchall(), int).flatten()
remove_items = setdiff1d(all_items, keep_items)

# Remove the items from the Item table
for item in remove_items:
    cursor.execute("DELETE FROM Item WHERE item_id=%s"%item)
connection.commit()

# Close cursor and connection
cursor.close()
connection.close()
