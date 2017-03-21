#!/usr/bin/env python
"""
Postgkyl test: testing gplot.py
"""
import subprocess
print('Postgkyl test: testing gplot.py')
try:
    subprocess.check_call(['python', '../postgkyl/gplot.py', '-p', 'data/data_bp_d_2_mo_p_4.bp', '--mo=4', '--dont'])
    print('PASSED: gplot.py tested successfully')
except subprocess.CalledProcessError as err:
    print('FAILED')
