"""
Create a database for the Food Balance Sheets Consumption and Production Data
Foreign keys: source_id, element_id, item_id, country_id
Make foreign key integer values all primary keys in their respective tables.
Create indices on each foreign key in the main data table.
"""
import numpy as np
from sqlite3 import dbapi2 as sqlite
import os
import sqlite_io

DB = r".\FAO_Database.db3"
N = np.arange(1961, 2011).size
connection = sqlite.connect(DB)
cursor = connection.cursor()

# Create new table to hold Commodity information
TABLE_NAME = "Commodity"
COLUMN_NAMES = """country_name, country_id, item_name, item_id, element_name, element_id, unit_id, source_id, """ + ", ".join("yr%s, yr%s_txt"%(i, i) for i in xrange(1961, 2011))
COLUMN_TYPES = """TEXT, INTEGER, TEXT, INTEGER, TEXT, INTEGER, INTEGER, INTEGER, """ + ", ".join(
    "%s, %s"%(i,j) for i, j in (zip(["FLOAT"] * N, ["TEXT"] * N))
)
Q = "CREATE TABLE %s (id INTEGER PRIMARY KEY AUTOINCREMENT, %s)"%(TABLE_NAME, ", ".join(
    "%s %s"%(i, j) for i, j in zip(COLUMN_NAMES.split(","), COLUMN_TYPES.split(","))
))
#cursor.execute(Q)
#connection.commit()

# Get Commodity data from CSV files
names = COLUMN_NAMES.split(",")
formats = [np.object, np.float, np.object, np.float, np.object, np.float, np.float, np.float] + [np.float, np.object] * N
dtype = zip(names, formats)
dir = "./FAO DATA - 5.23.12"
for files in os.walk(dir):
    csv_files = [file for file in files[2] if file.endswith(".csv")]

# Insert the data into the database
primary_key_count = 0
#for csv_file in csv_files:
#    xs = np.genfromtxt(os.path.join(dir, csv_file), delimiter="|", names=True, usemask=True, dtype=dtype)
#    sqlite_io.tosqlite(xs, primary_key_count, DB, TABLE_NAME, create=False, insert=True, autoid=True)
#    primary_key_count += xs.shape[0]

# Create new table to hold Demographic information
N = np.arange(1961, 2031).size
TABLE_NAME = "Demographic"

COLUMN_NAMES = """country_name, country_id, country_code, item_id, element_name, element_id, unit_id, source_id, """ + ", ".join("yr%s"%i for i in xrange(1961, 2031))
COLUMN_TYPES = """TEXT, INTEGER, TEXT, INTEGER, TEXT, INTEGER, INTEGER, INTEGER, """ + ", ".join(
    "%s"%i for i in ["FLOAT"] * N)
Q = "CREATE TABLE %s (id INTEGER PRIMARY KEY AUTOINCREMENT, %s)"%(TABLE_NAME, ", ".join(
    "%s %s"%(i, j) for i, j in zip(COLUMN_NAMES.split(","), COLUMN_TYPES.split(","))))

cursor.execute(Q)
connection.commit()

# Get Commodity data from CSV files
names = COLUMN_NAMES.split(",")
formats = [np.object, np.float, np.object, np.float, np.object, np.float, np.float, np.float] + [np.float] * N
dtype = zip(names, formats)
dir = "./Demographic Data - 5.23.12"
for files in os.walk(dir):
    csv_files = [file for file in files[2] if file.endswith(".csv")]

# Insert the data into the database
primary_key_count = 0
for csv_file in csv_files:
    xs = np.genfromtxt(os.path.join(dir, csv_file), delimiter="|", names=True, usemask=True, dtype=dtype)
#    sqlite_io.tosqlite(xs, primary_key_count, DB, TABLE_NAME, create=False, insert=True, autoid=True)

# Close cursor and connection
cursor.close()
connection.close()
