"""Testing the forecasting functions"""
import numpy as np
import apsw
import time
from optimized_Forecasting_Functions import expSmooth
from joblib import Memory
from tempfile import mkdtemp
cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, mmap_mode='r', verbose=0)
fCastPeriods = 20 #number of periods to forecast
f = expSmooth() #initialize exponential smoothing class

@memory.cache
def exp_smooth_forecast(xs, fCastPeriods, trend_modifier=.98):
    """Runs the exponential smoothing forecasting routine and returns the forecasted array.
    Stores previous calculated arrays in memory for easier retrieval and no unnecessary calculations.
    """
    print "entering the function"
    fitOpt = f.calcVariableArrays(trend_modifier, xs, fCastPeriods)
    return f.expSmoothForecast(fitOpt, True)

DB = r"C:\Users\calanoue\Dropbox\Code\res\sqlite_db.db3"
connection = apsw.Connection(DB)
cursor = connection.cursor()

Q = """
SELECT Prod FROM Commodities WHERE CCode=2555 AND (wbCode='USA' OR wbCode='BRA')
AND yr BETWEEN 1961 AND 2010
"""
desc = [('Prod', '<f8')]
xs = np.reshape(np.array(cursor.execute(Q).fetchall(), desc).view(float), (2, -1))
for row in xs:
    exp_forecast = exp_smooth_forecast(row, fCastPeriods)
cursor.close()
connection.close()


