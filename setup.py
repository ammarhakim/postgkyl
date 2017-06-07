#!/usr/bin/env python
from distutils.core import setup

try:
   import pypandoc
   long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   long_description = ''

setup(name='postgkyl',
      version='0.9.1',
      description='Postrocessing utilities for Gkyl simulation framework',
      long_description=long_description,
      url='https://bitbucket.org/ammarhakim/postgkyl',
      packages=['postgkyl',
                'postgkyl.data',
                'postgkyl.diagnostics',
                'postgkyl.tools',
                'postgkyl.commands'],
      include_package_data=True,
      data_files=[('data', ['postgkyl/data/xformMatricesModalMaximal.h5', 'postgkyl/data/xformMatricesModalSerendipity.h5', 'postgkyl/data/xformMatricesNodalSerendipity.h5', 'postgkyl/commands/postgkyl.mplstyle'])],
      entry_points='''
        [console_scripts]
        pgkyl=postgkyl.pgkyl:cli
      ''',
   )
