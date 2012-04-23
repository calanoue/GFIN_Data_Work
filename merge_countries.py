"""
Merge a country that has two separate former entities, i.e. Russia and the 'Slavias.
"""
import numpy as np
import sqlite_io
import numpy.lib.recfunctions as nprf


#Global variables
DB = r".\FBS_ProdSTAT_PriceSTAT_TradeSTAT.db3"
TABLE_NAME = "Commodity_Raw_Data"

#Countries to merge with country one being the country to stay in the database
country_one = 185
country_two = 228
country_name = "Russian_Federation" #table_name

#Query to merge the rows of the two countries
query = """
SELECT country_id, item_id, element_id, unit_id, source_id, %s
FROM %%s
WHERE country_id=%s OR country_id=%s
GROUP BY item_id||element_id||source_id
"""%(",".join("SUM(yr%s) AS yr%s"%(x, x) for x in xrange(1961, 2011)), country_one, country_two)

#Run query through sqlite_io file, creating a temporary table and then dropping when complete
xs = sqlite_io.fromsqlite(DB, query%TABLE_NAME, "tmp")
print xs[xs['item_id']==1012]
exit()

#Extract out merged data for country remaining in the database
xs_merged = xs[xs['country_id']==country_one]

#Create a new table in the database for this new merged country
count = 0
foreign_keys = {'country_id':'Country', 'element_id':'Element',
                'unit_id':'Unit', 'source_id':'Source'}
index = ['source_id', 'element_id', 'item_id', 'country_id'] #index in order
sqlite_io.tosqlite(xs_merged, count, DB, country_name, autoid=True,
    foreign_keys=foreign_keys, index=index)
