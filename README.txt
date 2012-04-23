Global Food in 3D Database Documentation:

Download Data:
 - Log into the FAO Stat system for the ability to do bulk downloads.
 - Food Balance Sheets (FBS): http://faostat.fao.org/site/614/default.aspx
 - Production STAT: http://faostat.fao.org/site/339/default.aspx
 - Price STAT: http://faostat.fao.org/site/703/default.aspx
 - Trade STAT: http://faostat.fao.org/site/406/default.aspx
 - Resource STAT: http://faostat.fao.org/site/550/default.aspx
 
Need to download the trade matrix in parts, maybe imports at one time and then exports 
 - Trade STAT (trade matrix): http://faostat.fao.org/site/537/default.aspx
 
Population and all GDP data from WorldBank
 - WorldBank: http://data.worldbank.org/

For the import stage, change delimiter to pipe delimited ("|") from comma delimited (",") because some of the item names have commas in them:
Control Panel -> Region and Language -> Formats -> Additional Settings -> List Separator = |
 
Format Data in CSV files:
Delete columns with formatting codes.
Find unique values for foreign keys and then replace text with integer id values following the below format
country_id, item_id, element_id, unit_id, source_id, 1961, ..., 2010, ..., 2050

Under GFIN_Data_Work:
 - Create foreign key and main data tables:
   - Run create_all_data_db.py to create database with foreign key tables and data tables with references and indexes to foreign key tables. 
     create_all_data_db uses sqlite_io.py to insert masked data into the database as <null> values. See FBS_ProdSTAT_PriceSTAT_TradeSTAT.db3 for
     an example of a finished product.
	 
 - Add in the SchemeColor table:
   - Create a new table and copy the SchemeColor table from the previous databases for the chloropleth map colors and data bins.
  
 - Create GFIN_DB database (which is used for the rest of the changes):
   - Run create_GFIN_DB.py to remove any rows of data that have less than five values and change masked values to -1, so that we don't need to use
     sqlite_io.py to get data from the database anymore. It is too slow for use in production because it makes many calls to the database to find
	 null values in each column and then masks those values. Also, adds Net Change and Per Capita elements to the Element table.
 
 - Aggregate commodities up to FBS item codes:
   - Run aggregate_commodities.py to aggregate commodity codes up to the FBS commodity code definitions in Commodity table. Production Price and Yield
     are aggregated by averaging the values, while all the rest are aggregated by summing. For the Production Quantity values, if the ProdSTAT values
	 exist, the FBS Production Quantity values will be removed.
	 
 - Delete item codes that no longer exist:
   - From the Item table, delete any items that are no longer found in the Commodity table by running delete_items_SQL.py
	 
 - Format and add TradeMatrix data:
   - Lookup trading countries in Country database table to find country_id values. Append either Import (Export) Quantity (Price) element code
     to the trading country id, i.e. 91_61 as the column headers.
   - Remove reporter, element, and items leaving only the columns with integer id values in the below format:
     reporting_country_id, year_id, item_id, 2_91, 2_92, ..., 181_91, 181_92
   - Run manipulate_trade_matrix.py to insert both import and export data into Trade table in the database with masked values as -1 values instead
     of <null> values. Make sure .\Commodity Code Conversions\CCode_to_ProdStat.csv is up-to-date with the commodity aggregations. Also, change
	 num_rows variable to a number larger than the number of countries * 4 elements * number of items.
	 
 - Insert Area Groupings table for sake of aggregating population values for per capita calculations
   - Run CREATE TABLE AreaGroup (id INTEGER PRIMARY KEY AUTOINCREMENT, group_id INTEGER REFERENCES Country, country_id INTEGER REFERENCES Country)
   - Get Area Groupings from http://faostat.fao.org/site/371/default.aspx and keep all countries that are found in the Country table. Check on China
     because its country_id could be 351 from this sheet but 357 in the database. See FaostatAreaGroupList.csv for an example.
	 
 - Calculate per capita variables:
   - Create new element_ids for Consumption (101) and Production per capita (52).
   - Create new unit_ids for <unique unit>/person.
     - SELECT DISTINCT unit_id FROM Commodity WHERE element_id=51 OR element_id=100
   - Create new source_id for GFIN calculated element.
   - Run per_capita_calculation.py to create new rows for Consumption and Production per capita with the above changes.
	 
 - Add location and color data to the Country Table:
   - Calculate the centroids of the regions.
   - Find new algorithms for assigning different colors to many different countries and regions. This website contains a couple suggestions:
     http://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors   
   - Run load_country_info.py to add x and z centroid values and R, G, and B color values to each country from the .\Country_Colors_Centroids.csv file.
    
 
