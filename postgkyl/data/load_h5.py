def load_h5(file_name : str, meta : dict) -> tuple:
    import tables
    fh = tables.open_file(file_name, 'r')
    if not '/StructGridField' in fh:
        fh.close()
        return False, ()
    #end
            
    # Get the atributes
    lower = fh.root.StructGrid._v_attrs.vsLowerBounds
    upper = fh.root.StructGrid._v_attrs.vsUpperBounds
    cells = fh.root.StructGrid._v_attrs.vsNumCells
    # Load data ...
    values = fh.root.StructGridField.read()
    # ... and the time-stamp
    if '/timeData' in fh:
        meta['time'] = fh.root.timeData._v_attrs.vsTime
    #end
    fh.close()
    return True, (lower, upper, cells, values)
#end
