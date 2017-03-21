#!/usr/bin/env python
"""
Postgkyl test: ADIOS files loading and projecting
"""
import numpy as np
import glob
import sys

import postgkyl as pg

print('Postgkyl test: ADIOS files loading and projecting')

files = glob.glob('data/*.bp')
if files == '':
    print('FAILURE: No test files found')
    sys.exit()

passedCnt = 0
for i, fileName in enumerate(files):
    sys.stdout.write(' *  Testing \'{:s}\' [{:d}/{:d}]'.format(fileName, i, len(files)))
    sys.stdout.flush()
    name = fileName.split('.')[-2] # get rid of the extension
    polyOrder = int(name.split('_')[-1])
    basis = name.split('_')[-3]
    numDims = int(name.split('_')[-4])

    try:
        data = pg.GData(fileName)

        if basis == 'ns':
            dg = pg.GInterpNodalSerendipity(data, polyOrder)
        elif basis == 'ms':
            dg = pg.GInterpModalSerendipity(data, polyOrder)
        elif basis == 'mo':
            dg = pg.GInterpModalMaxOrder(data, polyOrder)
        else:
            raise NameError('unsupported basis \'{:s}\'\n'.format(basis))

        coords, values = dg.project()

        # Check dimensions
        if coords.shape[0] != len(values.shape):
            raise RuntimeError('number of coordinates ({:d}) does not match the dimensions of values ({:d})\n'.format(coords.shape[0], len(values.shape)))

        if coords.shape[1 :] != values.shape:
            raise RuntimeError('dimensions of coordinates and values do not match\n')
        # Check values
        if values.max()-values.min() > 1e-13:
            raise RuntimeError('projected values do not match the initialization\n')
    except:
        sys.stdout.write('  FAILED - {:s}\n'.format(sys.exc_info()[1]))
        sys.stdout.flush()
        continue

    sys.stdout.write('  PASSED\n')
    sys.stdout.flush()
    passedCnt += 1

if passedCnt == len(files):
    print('PASSED: All files loaded and projected succesfully')
else:
    print('FAILURE: {:d} files failed'.format(len(files)-passedCnt))
