"""
SQL statements for creating a database
"""

# SQL statements for creating new tables in the new database
Q = """
CREATE TABLE Source (source_id INTEGER PRIMARY KEY, source_name TEXT, source_link TEXT);
CREATE TABLE Unit (unit_id INTEGER PRIMARY KEY, unit_name TEXT);
CREATE TABLE Year (id INTEGER PRIMARY KEY, year_id INTEGER, decade_id INTEGER);
CREATE TABLE SchemeColor (id INTEGER PRIMARY KEY AUTOINCREMENT, Scheme TEXT, DataClasses INTEGER, ColorNum INTEGER, R FLOAT, G FLOAT, B FLOAT);
CREATE TABLE Item (item_id INTEGER PRIMARY KEY, item_name TEXT);
CREATE TABLE Element (element_id INTEGER PRIMARY KEY, element_name TEXT, element_note TEXT, element_link TEXT);
CREATE TABLE Country (country_id INTEGER PRIMARY KEY, country_name TEXT, country_code TEXT, country_alt_name TEXT, x FLOAT, z FLOAT, R FLOAT, G FLOAT, B FLOAT);
CREATE TABLE AreaGroup (id INTEGER PRIMARY KEY AUTOINCREMENT, group_id INTEGER REFERENCES Country, country_id INTEGER REFERENCES Country);
"""

Q_index = """
CREATE INDEX source_index ON Source (source_id);
CREATE INDEX unit_index ON Unit (unit_id);
CREATE INDEX item_index ON Item (item_id);
CREATE INDEX element_index ON Element (element_id);
CREATE INDEX country_index ON Country (country_id);
"""

Q_distinct = """
SELECT DISTINCT element_id, element_name FROM Commodity
UNION ALL
SELECT DISTINCT element_id, element_name FROM Demographic
ORDER BY element_id

SELECT * FROM (SELECT DISTINCT country_id, country_name FROM
(SELECT DISTINCT country_id, country_name FROM Commodity
UNION ALL
SELECT DISTINCT country_id, country_name FROM Demographic
)) GROUP BY country_id ORDER BY country_id

SELECT DISTINCT item_id, item_name COLLATE nocase
FROM Commodity ORDER BY item_id DESC;

"""
# Remove feed, stock variation, SLC/tonne, LCU/tonne and other util
Q_deletions = """
DELETE FROM Commodity WHERE element_id=101;
DELETE FROM Commodity WHERE element_id=71;
DELETE FROM Commodity WHERE element_id=53 OR element_id=59;
DELETE FROM Commodity WHERE element_id=151;
"""

COLUMN_NAMES = """id, country_name, country_id, country_code, item_id, element_name, element_id, unit_id, source_id, """ + ", ".join("yr%s"%i for i in xrange(1961, 2031))
Q = """CREATE TABLE xs AS SELECT %s FROM Commodity;
DROP TABLE Commodity;"""%COLUMN_NAMES
print Q
