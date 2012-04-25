"""
Create Year table to hold all available years and their corresponding decades
"""
import numpy as np
from sqlite3 import dbapi2 as sqlite

# Connect to the database and create a cursor
DB = r".\GFIN_DB.db3"
connection = sqlite.connect(DB)
cursor = connection.cursor()

years = np.arange(1961, 2031) # possible years
decades = np.mod(np.mod(years, 1000), 100)/10*1. # corresponding decades
decades[np.where(decades==3)] = 2. # change 2030 to 2020 decade
ids = np.arange(years.size)*1. # primary key ids

# Array to insert into new table
xs = zip(ids, years*1., decades)

# Create table and insert values
create = "CREATE TABLE Year (id INTEGER PRIMARY KEY, year_id INTEGER, decade_id INTEGER)"
cursor.execute(create)
insert = "INSERT INTO Year VALUES (?,?,?)"
cursor.executemany(insert, xs)
connection.commit()

# Close the cursor and the connection
cursor.close()
connection.close()
