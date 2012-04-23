"""
Functions to help with data formatting and analysis.
"""

def get_dtype(connection, table, nameReturn=False, remove_id=False):
    """
    Get numpy data type from a database table
    """
    (names, typestr) = zip(*(_[1:3] for _ in connection.execute("PRAGMA TABLE_INFO(%s)"%table).fetchall()))

    # Get the data type for the specified table
    ndtype = []
    for n, t in zip(names, typestr):
        if t =='INTEGER':
            ndtype.append((n, float))
        elif t[:4]=='TEXT':
            ndtype.append((n, object)) # to handle large string values
        else:
            ndtype.append((n, float))

    # Remove field named id if requested
    if remove_id:
        names = [name.strip() for name in names if name!='id']
        ndtype =  ndtype[1:]

    # Return names if requested
    if nameReturn:
        return ndtype, names
    else:
        return ndtype

def mask_none_values(xs):
    """
    Mask None values that are actually <null> values in the database.
    """
    descr_names = xs.dtype.names # column names in the masked array
    for descr_name in descr_names:
        mask_idx = [enum for enum, j in enumerate(xs[descr_name]) if j==None] # index of None values
        xs[descr_name].mask[mask_idx] = True
    return xs

def add_to_element():
    """
    New elements for the Element table
    """
    new_elements = {
        52:"Production Quantity Per Capita", 101:"Consumption Quantity Per Capita", 102:"Consumption Elasticity",
        611:"Total Population - Both Sexes Net Change", 612:"Total Population - Male Net Change",
        613:"Total Population - Female Net Change", 651:"Rural population Net Change",
        661:"Urban population Net Change", 671:"Agricultural population Net Change",
        681:"Non-agricultural population Net Change", 691:"Total economically active population Net Change",
        692:"Male economically active population Net Change",
        693:"Female economically active population Net Change",
        701:"Total economically active population in Agr Net Change",
        702:"Male economically active population in Agr Net Change",
        703:"Female economically active population in Agr"
    }
    return new_elements
