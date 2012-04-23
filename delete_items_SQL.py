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

# Remove the items from the Item table that are no longer found in the Commodity table
cursor.execute("DELETE FROM Item WHERE (%s)"%" OR ".join("item_id=%s"%i for i in remove_items))
connection.commit()

# Close cursor and connection
cursor.close()
connection.close()
