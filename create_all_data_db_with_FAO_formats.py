"""
Create a database for the Food Balance Sheets Consumption and Production Data
Foreign keys: source_id, element_id, item_id, country_id
Make foreign key integer values all primary keys in their respective tables.
Create indices on each foreign key in the main data table.
"""
import numpy as np
import os
import sqlite_io

DB = r".\FAO_Database.db3"
TABLE_NAME = "Commodity"
N = np.arange(1961, 2031).size
connection = sqlite.connect(DB)
cursor = connection.cursor()

# Create new table to hold Commodity information
COLUMN_NAMES = """country_name, country_id, item_name, item_id, element_name, element_id, unit_id, source_id, """ + ", ".join("yr%s, yr%s_txt"%(i, i) for i in xrange(1961, 2031))
COLUMN_TYPES = """TEXT, INTEGER, TEXT, INTEGER, TEXT, INTEGER, INTEGER, INTEGER, """ + ", ".join(
    "%s, %s"%(i,j) for i, j in (zip(["FLOAT"] * N, ["TEXT"] * N))
)
Q = "CREATE TABLE %s (id INTEGER PRIMARY KEY AUTOINCREMENT, %s)"%(TABLE_NAME, ", ".join(
    "%s %s"%(i, j) for i, j in zip(COLUMN_NAMES.split(","), COLUMN_TYPES.split(","))
))

# Get Commodity data from CSV files
names = COLUMN_NAMES.split(",")
formats = [np.object, np.float, np.object, np.float, np.object, np.float, np.float, np.float] + [np.float, np.object] * N
dtype = zip(names, formats)
dir = "./FAO DATA - 5.23.12"
for files in os.walk(dir):
    csv_files = [file for file in files[2] if file.endswith(".csv")]
primary_key_count = 0
for csv_file in csv_files:
    if csv_file == "Prices.csv":
        xs = np.genfromtxt(os.path.join(dir, csv_file), delimiter="|", names=True, usemask=True, dtype=dtype)
        sqlite_io.tosqlite(xs, primary_key_count, DB, TABLE_NAME, create=False, insert=True)
        exit()


#Create separate tables with primary keys for the foreign keys in the main data table
cols = ['Country', 'Item', 'Element', 'Unit', 'Source']
#foreign_keys = {}
#for col in cols:
#    id_name = '%s_id'%col.lower()
#    dtype = ['<i4', '|S50']
#    xs = np.genfromtxt(".\%s.csv"%col, delimiter="|", names=True, usemask=True, dtype=dtype)
#    uniq_xs = np.unique(xs)
#    sqlite_io.tosqlite(uniq_xs, 0, DB, col, primary_key=[id_name])
#    foreign_keys[id_name] = col
#    print col, "Done"
#print "foreign keys", foreign_keys

#Create the main data table with foreign keys and proper indexes

#Get data from pipe delimited file - see control panel and regions for how to change this
#foreign_keys = {'item_id': 'Item', 'source_id': 'Source', 'element_id': 'Element', 'country_id': 'Country', 'unit_id': 'Unit'}
#dtype = ['<i4']*5 + ['<f8', '|S31']*50
#count = 472708
#for i in xrange(4, 5):
#    xs = np.genfromtxt(".\FBS_PriceSTAT_ProdSTAT_TradeSTAT-All_with_formats_%s.csv"%i,
#        delimiter="|", names=True, usemask=True, dtype=dtype)
#    create = [False, True][i==1]
#    foreign_keys = [[], foreign_keys][i==1]
#    sqlite_io.tosqlite(xs, count, DB, TABLE_NAME, autoid=True,
#        foreign_keys=foreign_keys, create=create)
#    count += xs.size
#    print "Main Data", i, "Done", "Count", count

#Add indexes to the main data table
count = 0
index = ['source_id', 'element_id', 'item_id', 'country_id'] #index in order
xs = np.genfromtxt(".\Source.csv", delimiter="|", names=True, usemask=True, dtype=None) #small xs
xs = np.unique(xs)
sqlite_io.tosqlite(xs, count, DB, TABLE_NAME, index=index, create=False, insert=False)

cursor.close()
connection.close()
