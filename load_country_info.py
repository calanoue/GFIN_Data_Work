"""
Load country centroid and color data
"""

import sqlite3
import numpy as np

# Connect to the database and create a cursor for extracting data
DB = r".\FBS_ProdSTAT_PriceSTAT_TradeSTAT.db3"
connection = sqlite3.connect(DB)
cursor = connection.cursor()

xs = np.genfromtxt(".\Country_Colors_Centroids.csv",
    delimiter=",", names=True, usemask=True, dtype=None)
for y in xs:
    x, z, R, G, B, id = y
    Q = "UPDATE Country SET x=%s, z=%s, R=%s, G=%s, B=%s WHERE country_id=%s"%(x, z, R, G, B, id)
    cursor.execute(Q)
connection.commit()

cursor.close()
connection.close()
