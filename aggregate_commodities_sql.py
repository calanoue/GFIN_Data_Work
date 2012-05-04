"""
Aggregate commodities based on their FAO aggregation relationships.
"""
import numpy as np
import sqlite3
from get_numpy_dtype import format_creates, format_indexes
from itertools import izip

# Connect to the database and create a cursor for extracting data
DB = r".\FBS_ProdSTAT_PriceSTAT_TradeSTAT.db3"
TABLE_NAME = "Commodity"
connection = sqlite3.connect(DB)
cursor = connection.cursor()

def format_sql_str(statement):
    """
    Remove brackets and carriage returns from a SQL statement
    """
    replace_strs = ["]", "[", "\r\n"]
    for replace_str in replace_strs:
        statement = statement.replace(replace_str, "")
    return statement

# Get create statement for new table from DDL
Q = 'SELECT sql FROM sqlite_master WHERE type="table" AND tbl_name="%s"'%TABLE_NAME
create_statement, = np.array(cursor.execute(Q).fetchall()).flatten()
create_statement = format_sql_str(create_statement)

# Get index statement for new table from DDL
Q = 'SELECT tbl_name, sql FROM sqlite_master WHERE type="index" AND tbl_name="%s"'%TABLE_NAME
index_statement = np.array(cursor.execute(Q).fetchall())
index_statement = format_indexes(index_statement)[0]

# Get all item codes that need to be aggregated into another commodity
ys = np.genfromtxt(".\Commodity Code Conversions\CCode_to_ProdStat.csv", delimiter=",", usemask=True, dtype=None)
unique_aggregate_items = np.unique(ys) # items that need to be summed, averaged, etc
aggregate_item_lookup = dict(izip(ys[:, 0], ys[:, 1:])) # dictionary of item codes, {main_item_id:aggregate_item_ids}

# Query columns
id_names = ["id", "country_id", "item_id", "element_id", "unit_id", "source_id"]
years = np.arange(1961, 2011)
names =  id_names + ["yr%s"%x for x in years]
sum_names = id_names + ["SUM(yr%s) AS yr%s"%(x, x) for x in years]
avg_names = id_names + ["AVG(yr%s) AS yr%s"%(x, x) for x in years]

# Create a table to handle the aggregation results
cursor.execute("""CREATE TABLE commodity_xs AS SELECT %s FROM %s WHERE 1=2"""%(",".join(names[1:]), TABLE_NAME))
connection.commit()

# Main Query String that aggregates the values
Q = """
CREATE TEMP TABLE commodity_tmp AS
SELECT %s FROM Commodity WHERE (%%s) AND element_id NOT IN (41, 60) GROUP BY country_id||element_id
UNION ALL
SELECT %s FROM Commodity WHERE (%%s) AND (element_id=41 OR element_id=60) GROUP BY country_id||element_id
UNION ALL
SELECT %s FROM Commodity WHERE item_id=%%s;
CREATE TEMP TABLE item_where AS SELECT id, MIN(item_id) FROM commodity_tmp GROUP BY country_id||element_id;
INSERT INTO commodity_xs
SELECT %%s FROM commodity_tmp INNER JOIN item_where ON commodity_tmp.id=item_where.id;
DROP TABLE item_where; DROP TABLE commodity_tmp;
"""%(",".join(sum_names), ",".join(avg_names), ",".join(names))

# Go through each key and value array in the aggregate dictionary and sum/average the arrays
for key, item_ids in aggregate_item_lookup.iteritems():
    item_N = np.transpose(np.ma.notmasked_edges(item_ids, axis=1))[1] + 1 # number of items to aggregate
    item_id_str = " OR ".join("item_id=%s"%v for v in item_ids[:item_N])

    # Format column names list to remove id column and change item_id values to the key value
    names[2] = "%s AS item_id"%key
    cursor.executescript(Q%(item_id_str, item_id_str, key, ",".join(names[1:])))

# Get item_ids that don't need to be aggregated
unique_items = np.array(cursor.execute("SELECT item_id FROM Item").fetchall()).flatten()
keep_items_SQL = """INSERT INTO commodity_xs SELECT %s FROM %s WHERE %s"""%(
    ",".join(names[1:]), TABLE_NAME, " OR ".join(
        "item_id=%s"%item_j for item_j in [item_i for item_i in unique_items if item_i not in unique_aggregate_items])
    )
cursor.execute(keep_items_SQL)
connection.commit()

# Drop old table and create new table with aggregated data
cursor.execute("DROP TABLE %s"%TABLE_NAME)
connection.commit()
cursor.execute(create_statement)
connection.commit()

# Insert data into the new Commodity table after sorting by index column names and drop temp table
Q = """
INSERT INTO %s SELECT rowid, * FROM commodity_xs ORDER BY source_id, element_id, item_id, country_id;
DROP TABLE commodity_xs;
"""%TABLE_NAME
cursor.executescript(Q)
connection.commit()

# Create index on the table
cursor.execute(index_statement)
connection.commit()

# Close cursor and connection
cursor.close()
connection.close()