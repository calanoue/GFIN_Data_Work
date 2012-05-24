"""
Template file for creating a connection to the database and then doing something.
"""
import numpy as np
from sqlite3 import dbapi2 as sqlite
from itertools import imap

# Connect to the database and create a cursor
DB = r".\GFIN_DB.db3"
connection = sqlite.connect(DB)
connection.text_factory = str # use 8 bit strings instead of unicode strings in SQLite
cursor = connection.cursor()

# Get demographic values
select = """
SELECT country_id,-1 AS item_id,element_id,unit_id,source_id,yr1961,yr1962,yr1963,yr1964,yr1965,
yr1966,yr1967,yr1968,yr1969,yr1970,yr1971,yr1972,yr1973,yr1974,yr1975,yr1976,yr1977,yr1978,yr1979,yr1980,
yr1981,yr1982,yr1983,yr1984,yr1985,yr1986,yr1987,yr1988,yr1989,yr1990,yr1991,yr1992,yr1993,yr1994,yr1995,
yr1996,yr1997,yr1998,yr1999,yr2000,yr2001,yr2002,yr2003,yr2004,yr2005,yr2006,yr2007,yr2008,yr2009,yr2010,
yr2011,yr2012,yr2013,yr2014,yr2015,yr2016,yr2017,yr2018,yr2019,yr2020,yr2021,yr2022,yr2023,yr2024,yr2025,
yr2026,yr2027,yr2028,yr2029,yr2030
FROM Demographic
"""
xs = np.array(cursor.execute(select).fetchall())

# Get commodity values
select = """
SELECT country_id,item_id,element_id,unit_id,source_id,yr1961,yr1962,yr1963,yr1964,yr1965,
yr1966,yr1967,yr1968,yr1969,yr1970,yr1971,yr1972,yr1973,yr1974,yr1975,yr1976,yr1977,yr1978,yr1979,yr1980,
yr1981,yr1982,yr1983,yr1984,yr1985,yr1986,yr1987,yr1988,yr1989,yr1990,yr1991,yr1992,yr1993,yr1994,yr1995,
yr1996,yr1997,yr1998,yr1999,yr2000,yr2001,yr2002,yr2003,yr2004,yr2005,yr2006,yr2007,yr2008,yr2009,yr2010,
-1 AS yr2011,-1 AS yr2012,-1 AS yr2013,-1 AS yr2014,-1 AS yr2015,-1 AS yr2016,-1 AS yr2017,-1 AS yr2018,
-1 AS yr2019,-1 AS yr2020,-1 AS yr2021,-1 AS yr2022,-1 AS yr2023,-1 AS yr2024,-1 AS yr2025,
-1 AS yr2026,-1 AS yr2027,-1 AS yr2028,-1 AS yr2029,-1 AS yr2030
FROM Commodity
"""
ys = np.array(cursor.execute(select).fetchall())

# Stack the two arrays
zs = np.vstack((ys, xs))

# Create new table
create = """
CREATE TABLE Datum (id INTEGER PRIMARY KEY AUTOINCREMENT,country_id INTEGER REFERENCES Country,
item_id INTEGER REFERENCES Item,element_id INTEGER REFERENCES Element,unit_id INTEGER REFERENCES Unit,
source_id INTEGER REFERENCES Source,yr1961 FLOAT,yr1962 FLOAT,yr1963 FLOAT,yr1964 FLOAT,yr1965 FLOAT,
yr1966 FLOAT,yr1967 FLOAT,yr1968 FLOAT,yr1969 FLOAT,yr1970 FLOAT,yr1971 FLOAT,yr1972 FLOAT,yr1973 FLOAT,
yr1974 FLOAT,yr1975 FLOAT,yr1976 FLOAT,yr1977 FLOAT,yr1978 FLOAT,yr1979 FLOAT,yr1980 FLOAT,yr1981 FLOAT,
yr1982 FLOAT,yr1983 FLOAT,yr1984 FLOAT,yr1985 FLOAT,yr1986 FLOAT,yr1987 FLOAT,yr1988 FLOAT,yr1989 FLOAT,
yr1990 FLOAT,yr1991 FLOAT,yr1992 FLOAT,yr1993 FLOAT,yr1994 FLOAT,yr1995 FLOAT,yr1996 FLOAT,yr1997 FLOAT,
yr1998 FLOAT,yr1999 FLOAT,yr2000 FLOAT,yr2001 FLOAT,yr2002 FLOAT,yr2003 FLOAT,yr2004 FLOAT,yr2005 FLOAT,
yr2006 FLOAT,yr2007 FLOAT,yr2008 FLOAT,yr2009 FLOAT,yr2010 FLOAT,yr2011 FLOAT,yr2012 FLOAT,yr2013 FLOAT,
yr2014 FLOAT,yr2015 FLOAT,yr2016 FLOAT,yr2017 FLOAT,yr2018 FLOAT,yr2019 FLOAT,yr2020 FLOAT,yr2021 FLOAT,
yr2022 FLOAT,yr2023 FLOAT,yr2024 FLOAT,yr2025 FLOAT,yr2026 FLOAT,yr2027 FLOAT,yr2028 FLOAT,yr2029 FLOAT,
yr2030 FLOAT)
"""
cursor.execute(create)
connection.commit()

# Get data type for datum table
top_splits = create.split(",")[1:]
for i, top in enumerate(top_splits):
    splits = top.split(" ")[:2]
    for j, split in enumerate(splits):
        splits[j] = split.replace("\n", "").replace(")", "").replace("FLOAT", "<f8").replace("INTEGER", "<f8")
    top_splits[i] = splits
ndtype = [i for i in imap(tuple, top_splits)]

# View combined data as a recarray
zs = zs.view(ndtype).flatten()

# Sort new array by index fields in order
zs = np.sort(zs, order=['element_id', 'item_id', 'country_id'])

# Insert data into the new table
import sqlite_io
sqlite_io.tosqlite(zs, 0, DB, "Datum", autoid=True, create=False)

# Add on index
index = "CREATE INDEX Datum_index ON Datum (element_id, item_id, country_id)"
cursor.execute(index)
connection.commit()

# Drop Demographic and Commodity table
drops = ["DROP TABLE Demographic", "DROP TABLE Commodity"]
[cursor.execute(drop) for drop in drops]
connection.commit()

# Close the cursor and the connection
cursor.close()
connection.close()