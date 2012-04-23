"""
Update data
"""
import numpy as np
from sqlite3 import dbapi2 as sqlite

DB = r".\FBS_ProdSTAT_PriceSTAT_TradeSTAT.db3"
connection = sqlite.connect(DB)
cursor = connection.cursor()
TABLE_NAME = "Country"

xs = np.genfromtxt(".\country_upload.csv", delimiter="|", names=True, usemask=True, dtype=None)
Q = "UPDATE %s SET %%s WHERE country_id=%%s"%TABLE_NAME
for row in xs:
    code, name, id = row['country_code'], row['country_alt_name'], row['country_id']
    if name is np.ma.masked and code is np.ma.masked:
        continue
    elif name is np.ma.masked:
        query = Q%("country_code='%s'"%code, id)
    else:
        query = Q%("country_code='%s', country_alt_name='%s'"%(code, name), id)
#    cursor.execute(query)
#connection.commit()

cursor.close()
connection.close()