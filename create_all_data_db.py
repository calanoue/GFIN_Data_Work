"""
Create a database for the Food Balance Sheets Consumption and Production Data
Foreign keys: source_id, element_id, item_id, country_id
Make foreign key integer values all primary keys in their respective tables.
Create indices on each foreign key in the main data table.
"""
import numpy as np
import sqlite_io

DB = r".\FBS_ProdSTAT_PriceSTAT_TradeSTAT.db3"
#TABLE_NAME = "Commodity_Raw_Data"
TABLE_NAME = "Population_GDP_Raw_Data"

# Get data from pipe delimited file - see control panel and regions for how to change this
#xs = np.genfromtxt(".\FBS_PriceSTAT_ProdSTAT_TradeSTAT-All_with_formats.csv",
#    delimiter="|", names=True, usemask=True, dtype=None
# ) # read in all Commodity masking blank values
xs = np.genfromtxt(".\GDP-WorldBank - 3_19_12\WorldBank GDP Data - All.csv",
    delimiter="|", names=True, usemask=True, dtype=None
) # read in GDP and Population data masking blank values

# Create separate tables with primary keys for the foreign keys in the main data table
#cols = ['Country', 'Item', 'Element', 'Unit', 'Source']
#foreign_keys = {}
#for col in cols:
#    id_name = '%s_id'%col.lower()
#    uniq_vals = np.unique(xs[[id_name, '%sName'%col]])
#    sqlite_io.tosqlite(uniq_vals, DB, col, primary_key=[id_name])
#    foreign_keys[id_name] = col

# Columns to remove from main data because located in other columns
#xs = xs[[name for name in xs.dtype.names if name not in ['%sName'%col for col in cols]]]

# Create the main data table with foreign keys and proper indexes
#foreign_keys = {'country_id':'Country', 'item_id':'Item',
#                'element_id':'Element', 'unit_id':'Unit', 'source_id':'Source'
# } # foreign keys for Commodity table
foreign_keys = {'country_id':'Country', 'element_id':'Element',
                'unit_id':'Unit', 'source_id':'Source'
} # foreign keys for Population_GDP table
#index = ['source_id', 'element_id', 'item_id', 'country_id'] # index in order
index = ['source_id', 'element_id', 'country_id'] # index in order
count = 3575 # first record for insertion; start at 0 for new tables
#sqlite_io.tosqlite(xs, count, DB, TABLE_NAME, autoid=True,
#    foreign_keys=foreign_keys, index=index, create=False)