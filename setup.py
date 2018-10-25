#!/usr/bin/env python
from distutils.core import setup

try:
   import pypandoc
   long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   long_description = ''

setup(name='postgkyl',
      version='1.1.2',
      description='Postrocessing utilities for Gkyl simulation framework',
      long_description=long_description,
      url='https://bitbucket.org/ammarhakim/postgkyl',
      packages=['postgkyl',
                'postgkyl.data',
                'postgkyl.diagnostics',
                'postgkyl.output',
                'postgkyl.tools',
                'postgkyl.commands',
                'postgkyl.utils'],
      package_data={'': ['data/*.h5',
                         'output/postgkyl.mplstyle']},
      include_package_data=True,
      entry_points='''
        [console_scripts]
        pgkyl=postgkyl.pgkyl:cli
      ''',
   )
