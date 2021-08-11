#!/usr/bin/env python
from setuptools import setup

try:
   import pypandoc
   long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   long_description = ''

setup(name='postgkyl',
      version='1.6.6',
      description='Postrocessing utilities for Gkeyll simulation framework',
      long_description=long_description,
      url='https://github.com/ammarhakim/postgkyl',
      packages=['postgkyl',
                'postgkyl.data',
                'postgkyl.diagnostics',
                'postgkyl.output',
                'postgkyl.tools',
                'postgkyl.commands',
                'postgkyl.utils',
                'postgkyl.modalDG',
                'postgkyl.modalDG.kernels'],
      package_data={'': ['data/*.h5',
                         'output/postgkyl.mplstyle']},
      include_package_data=True,
      entry_points='''
        [console_scripts]
        pgkyl=postgkyl.pgkyl:cli
      ''',
   )
