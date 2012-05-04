"""
See: http://svn.scipy.org/svn/scikits/branches/pierregm/hydroclimpy/scikits/hydroclimpy/io/sqlite.py
Tools to connect ndarrays, MaskedArrays and TimeSeries with sqlite3 databases

Notes
-----

* sqlite3 doesn't recognize numpy types natively.
  One option is to register a set of converters with  a command like::

  sqlite3.register_adapter(np.int_, int)

  This command is called each time sqlite3 encounters a numpy object.
  It looked more efficient to transform a ndarray into a collection of Python
  objects using a ``np.object`` dtype.
  An advantage of this approach is that the missing values of a  MaskedArray can
  easily be converted into ``None`` (which represents a missing field in sqlite).
"""
from itertools import imap
import sqlite3
import numpy as np
import numpy.ma as ma

PARSE_DECLTYPES = sqlite3.PARSE_DECLTYPES
PARSE_COLNAMES = sqlite3.PARSE_COLNAMES
DETECT_TYPES = PARSE_DECLTYPES | PARSE_COLNAMES

DTYPE_TRANSLATOR = {
    'b': 'NUMERIC', 'i': 'INTEGER', 'l': 'INTEGER', 'u': 'FLOAT',
    'S': 'TEXT', 'U': 'TEXT', '?': 'FLOAT', 'd': 'FLOAT', 'O': 'TEXT'}

connect = sqlite3.connect

def _getsqldesc(marray):
    """
    Private function returning a list of tuples (name, SQLite type) for an input array.

    Parameters
    ----------
    marray : array
        Input array
    """
    mdtype = marray.dtype
    names = mdtype.names
    dtypechars = [mdtype[n].char for n in names]
    return ["%s %s" % (n, DTYPE_TRANSLATOR[c]) for (n, c) in zip(names, dtypechars)]

def _executesql(dbname, tablename, fieldlist, generator, index=[],
                overwrite=False, create=True, insert=True):
    """
    Private function creating and filling a table in the given database.

    Parameters
    ----------
    dbname: string
        Name of the database.
    tablename: string
        Name of the table to create.
    fieldlist : list
        List of tuples (name, SQL type) for each field of the table.
    generator : generator
        Generator of records.
    index : list
        List of fields for creating an index on the table
    overwrite: {False, True}
        Whether to overwrite an existing table or not.
        If not (default), an :exc:`sqlite3.OperationalError` exception is raised.
    create: {True, False}
        Whether to create a new table
    insert: {True, False}
        Where to insert the data into the table specified
    """
    
    # Connect w/ a database
    connection = sqlite3.connect(dbname, detect_types=DETECT_TYPES)
    connection.text_factory = str #use 8 bit strings instead of unicode strings in SQLite
    cursor = connection.cursor()
    
    # Create the table (overwrite if needed)
    if create:
        createstr = "CREATE TABLE %s (%s)" % (tablename, ", ".join(fieldlist)) # Define the creation line
        try:
            cursor.execute(createstr)
        except sqlite3.OperationalError:
            
            # Existing table: should we overwrite ?
            if overwrite:
                dropstr = "DROP TABLE %s"%tablename
                try:
                    cursor.execute(dropstr)
                except sqlite3.OperationalError, msg:
                    errmsg = "Could not overwrite table '%s': %s" % (tablename, msg)
                    raise sqlite3.OperationalError(errmsg)
                cursor.execute(createstr)
            else:
                raise
    
    # Fill the table
    if insert:
        insert_template = "INSERT INTO %s VALUES (%s)" # Define a template line for insertion
        insertstr = insert_template % (tablename, ", ".join(['?']*len(fieldlist)))
        try:
            cursor.executemany(insertstr, generator)
        except:
            raise
    
    # Create indexes if needed
    if index:
        indexstr = "CREATE INDEX %s_index ON %s(%s)"%(tablename.lower(), tablename,
                                                      ", ".join(i for i in index))
        cursor.execute(indexstr)
    connection.commit()
    connection.close()
    return connection

def _getdatafromsql(connection, tmp_table, query):
    """
    Private function creating a ndarray from the current table.

    Parameters
    ----------
    connection: sqlite3.Connection
        Current SQL connection.
    tmp_table: string
        Name of the temporary table created for the purpose of keeping ids when WHERE is used
    query: string
        SQL query.
    """
    # Transforms the typestr into dtypes
    
    # Define and execute the query
    connection.execute("CREATE TEMPORARY TABLE %s AS %s"%(tmp_table, query))
    
    # Get the list of names and types from the pragma
    pragmastr = "PRAGMA TABLE_INFO(%s)"%tmp_table
    (names, typestr) = zip(*(_[1:3] for _ in connection.execute(pragmastr).fetchall()))
    ndtype = []
    for (i, (n, t)) in enumerate(zip(names, typestr)):
        
        # Transform the name into a regular string (not unicode)
        n = str(n)
        if t =='INTEGER':
            ndtype.append((n, int))
        elif t =='TEXT':
            ndtype.append((n, '|S30'))
        elif t == 'BLOB':
            ndtype.append((n, object))
        else:
            ndtype.append((n, float))
    
    # Construct the ndarray
    connection.row_factory = sqlite3.Row
    data = connection.execute("SELECT * FROM %s"%tmp_table).fetchall()
    try:
        return np.array(data, dtype=ndtype)
    except TypeError:
        output = ma.empty(len(data), dtype=ndtype)
        
        # Find the index of the first row (0 or 1)?
        rowidref = connection.execute("SELECT rowid FROM %s LIMIT 1"%tmp_table).fetchone()[0]
        
        # Loop through the different fields identifying the null fields to mask
        maskstr_template = "SELECT rowid FROM %s WHERE %%s IS NULL"%tmp_table
        datastr_template = "SELECT %%s FROM %s WHERE %%s IS NOT NULL"%tmp_table
        for (i, field) in enumerate(names):
            current_output = output[field]
            current_mask = current_output._mask
            maskstr = maskstr_template % field
            maskidx = [_[0] - rowidref for _ in connection.execute(maskstr).fetchall()]
            current_mask[maskidx] = True
            datastr = datastr_template % (field, field)
            np.place(current_output._data, ~current_mask,
                [_[0] for _ in connection.execute(datastr).fetchall()])
        connection.execute("DROP TABLE %s"%tmp_table)
        return output

def tosqlite(series, count, dbname, tablename, overwrite=False, autoid=False,
             foreign_keys={}, primary_key=[], index=[], create=True, insert=True):
    """
    Save a MaskedArray to a SQLite table.

    The names of the fields (columns) are read from the series dtype.

    Parameters
    ----------
    series : TimeSeries
        Input data
    dbname: string
        Name of the database.
    tablename: string
        Name of the table to create.
    overwrite: {False, True}
        Whether to overwrite an existing table or not.
        If not (default), an :exc:`sqlite3.OperationalError` exception is raised.
    autoid: {False, True}
        Whether to create an autoincremented field 'id' to use as primary key
    foreign_keys: dictionary
        Columns that have foreign key constraints.
    primary_key: list
        If the primary key is not the auto incremented id, then use this field for the primary key
    index : list
        List of fields for creating an index on the table
    create: {True, False}
        Whether to create a new table
    insert: {True, False}
        Where to insert the data into the table specified
    """
    # Make sure we have an array
    a = series
    a = np.asanyarray(a)
    adtype = a.dtype
    a = a.ravel()
    fieldlist = _getsqldesc(a) # Get the list of fields
    ndtype = [(_, np.object) for _ in adtype.names] # Transform to a ndarray of Python objects
    a = a.astype(ndtype)
    
    # Create a generator
    if isinstance(a, ma.MaskedArray):
        a = a.filled([None] * len(adtype.names))
    if autoid:
        fieldlist.insert(0, "id INTEGER PRIMARY KEY AUTOINCREMENT")
        generator = (tuple([i] + list(_)) for (i, _) in enumerate(a, count))
    else:
        generator = imap(tuple, a)
    
    # Insert foreign keys into table
    if foreign_keys:
        for i, field in enumerate(fieldlist):
            name = field.split()
            if name[0] in foreign_keys.iterkeys():
                field += " REFERENCES %s"%foreign_keys[name[0]]
                fieldlist[i] = field
    
    # Insert primary key into table
    if primary_key:
        for i, field in enumerate(fieldlist):
            name = field.split()
            if name[0] in primary_key:
                field += " PRIMARY KEY"
                fieldlist[i] = field
    result = _executesql(dbname, tablename, fieldlist, generator, index, overwrite, create, insert)
    return result

def fromsqlite(dbname, query, tmp_table):
    """
    Create a MaskedArray from a SQLite table

    Parameters
    ----------
    dbname: string
        Name of the database for connecting.
    query: string
        SQL query
    tmp_table: string
        Name of the temporary table created for the purpose of keeping ids when WHERE is used
    """
    connection = sqlite3.connect(dbname, detect_types=DETECT_TYPES)
    try:
        return _getdatafromsql(connection, tmp_table, query)
    finally:
        connection.close()