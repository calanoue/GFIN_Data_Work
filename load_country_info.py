"""
Load country centroid and color data
"""

import sqlite3
import numpy as np

# Connect to the database and create a cursor for extracting data
DB = r"C:\Users\calanoue\Dropbox\Dont Want to Lose\GFIN Random Python Work\demoGFIN\sqlite_student_db.db3"
connection = sqlite3.connect(DB)
cursor = connection.cursor()

xs = np.genfromtxt(
    ".\Country_Colors_Centroids_from_old_database.csv", delimiter=",", names=True, usemask=True, dtype=None
)
Q = ""
for y in xs:
    wb_name, x, z, R, G, B, id = y
    Q += "UPDATE Country SET x=%s, z=%s, R=%s, G=%s, B=%s WHERE country_id=%s;\n"%(x, z, R, G, B, id)
cursor.executescript(Q)
connection.commit()

cursor.close()
connection.close()
