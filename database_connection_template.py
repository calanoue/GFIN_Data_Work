"""
Template file for creating a connection to the database and then doing something.
"""
import numpy as np
from sqlite3 import dbapi2 as sqlite

# Connect to the database and create a cursor
DB = r".\GFIN_DB.db3"
connection = sqlite.connect(DB)
connection.text_factory = str #use 8 bit strings instead of unicode strings in SQLite
cursor = connection.cursor()

# Do something
demo_names = {str(_[1]) for _ in connection.execute("PRAGMA TABLE_INFO(Demographic)").fetchall()}
print demo_names
commodity_names ={str(_[1]) for _ in connection.execute("PRAGMA TABLE_INFO(Commodity)").fetchall()}
demo_nulls = ",".join("-1 AS %s"%i for i in commodity_names - demo_names)
commodity_nulls = ",".join("-1 AS %s"%i for i in sorted(demo_names - commodity_names))
common_columns = sorted(demo_names.intersection(commodity_names))
print common_columns

# Close the cursor and the connection
cursor.close()
connection.close()