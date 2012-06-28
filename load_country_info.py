"""
Load country centroid and color data
"""

import sqlite3
import numpy as np

# Connect to the database and create a cursor for extracting data
DB = r"C:\Users\calanoue\Dropbox\Dont Want to Lose\GFIN Random Python Work\demoGFIN\sqlite_student_db.db3"
connection = sqlite3.connect(DB)
cursor = connection.cursor()

# Task 1: Update centroids and colors from old database
xs = np.genfromtxt(
    ".\Country_Colors_Centroids_from_old_database.csv", delimiter=",", names=True, usemask=True, dtype=None
)
Q = ""
for y in xs:
    wb_name, x, z, R, G, B, id = y
    Q += "UPDATE Country SET x=%s, z=%s, R=%s, G=%s, B=%s WHERE country_id=%s;\n"%(x, z, R, G, B, id)

# Task 2: Change countries not listed in this file to -1 x and z values
all_countries = np.array(
    cursor.execute("SELECT country_id FROM Country WHERE country_id < 5000").fetchall(), np.int
).flatten()
update_countries = " OR ".join("country_id=%s"%x for x in np.setdiff1d(all_countries, xs['country_id']))
Q = "UPDATE Country SET x=-1, z=-1, R=0, G=0, B=0 WHERE %s;"%update_countries

# Task 3: Change Democratic Republic of the Congo (id=250) to country_code ZAR
Q = "UPDATE Country SET country_code='ZAR' WHERE country_id=250;"

# Task 4: Update -1 x and z value countries to manually calculated centroids
xs = np.genfromtxt(
    ".\Manual_Centroid_Calculation.csv", delimiter=",", names=True, usemask=True, dtype=None
)
Q = ""
for y in xs:
    wb_name, wb_code, id, x, z = y
    Q += "UPDATE Country SET x=%s, z=%s WHERE country_id=%s;\n"%(x, z, id)

cursor.executescript(Q)
connection.commit()

cursor.close()
connection.close()
