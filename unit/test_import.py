#!/usr/bin/env python
"""
Postgkyl test: importing Postgkyl
"""
print('Postgkyl test: importing Postgkyl')
try:
    import postgkyl
    print('PASSED: Postgkyl imported successfully')
except:
    print('FAILED: Postgkyl was not imported')
