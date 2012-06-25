"""
Aggregate commodities based on their FAO aggregation relationships.
"""
import numpy as np
import sqlite3

# Connect to the database and create a cursor for extracting data
DB = r"C:\Users\calanoue\Dropbox\Dont Want to Lose\GFIN Random Python Work\demoGFIN\sqlite_student_db.db3"
connection = sqlite3.connect(DB)
cursor = connection.cursor()

def format_sql_str(statement):
    """
    Remove brackets and carriage returns from a SQL statement
    """
    replace_strs = ["]", "[a", "\r\n"]
    for replace_str in replace_strs:
        statement = statement.replace(replace_str, "")
    return statement

# Get create statement for new table from DDL
Q = 'SELECT sql FROM sqlite_master WHERE type="table" AND tbl_name="Datum"'
create_statement, = np.array(cursor.execute(Q).fetchall()).flatten()
create_statement = format_sql_str(create_statement)

# All item codes that need to be aggregated into another commodity
aggregate_item_lookup = {2581:36, 2594:332, 2590:238, 2595:259, 2592:269, 2562:256}

# TODO - remove some items from the Item table
#cursor.execute("SELECT DISTINCT item_id FROM Datum WHERE item_id!=-1")

# Query columns
id_names = ["id", "country_id", "item_id", "element_id", "unit_id", "source_id"]
names =  id_names + ["yr%s"%x for x in np.arange(1961, 2031)]

# Create a table to handle the aggregation results
cursor.execute("CREATE TABLE datum_xs AS SELECT %s FROM Datum WHERE 1=2"%",".join(names[1:]))
connection.commit()

# Main query string for aggregation
Q = """
CREATE TEMP TABLE datum_tmp AS
SELECT %s FROM Datum WHERE item_id=%%s OR item_id=%%s;
CREATE TEMP TABLE item_where AS SELECT id, MIN(item_id) FROM datum_tmp GROUP BY country_id||element_id;
INSERT INTO datum_xs
SELECT %%s FROM datum_tmp INNER JOIN item_where ON datum_tmp.id=item_where.id;
DROP TABLE item_where; DROP TABLE datum_tmp;
"""%",".join(names)

# Go through each key and value array in the aggregate dictionary
for key, item_id in aggregate_item_lookup.iteritems():
    item_id_str = "item_id=%s"%item_id

    # Format column names list to remove id column and change item_id values to the key value
    names[2] = "%s AS item_id"%key
    cursor.executescript(Q%(item_id_str, key, ",".join(names[1:])))

# Get item_ids that don't need to be aggregated
unique_items = np.array(cursor.execute("SELECT item_id FROM Item").fetchall()).flatten()
aggregate_items = np.array(aggregate_item_lookup.items()).flatten()
names[2] = "item_id"
keep_items_SQL = """INSERT INTO datum_xs SELECT %s FROM Datum WHERE %s OR item_id=-1"""%(
    ",".join(names[1:]), " OR ".join("item_id=%s"%item for item in np.setdiff1d(unique_items, aggregate_items))
    )
cursor.execute(keep_items_SQL)
connection.commit()

# Drop old table and create new table with aggregated data
cursor.execute("DROP TABLE Datum")
connection.commit()
cursor.execute(create_statement)
connection.commit()

# Insert data into the new Commodity table after sorting by index column names and drop temp table
Q = """INSERT INTO Datum SELECT rowid, * FROM datum_xs ORDER BY element_id, item_id, country_id;
DROP TABLE datum_xs;"""
cursor.executescript(Q)
connection.commit()

# Create index on Datum table
cursor.execute("CREATE INDEX Datum_index ON Datum (element_id, item_id, country_id)")
connection.commit()

# Close cursor and connection
cursor.close()
connection.close()
