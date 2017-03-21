#!/usr/bin/env python
from distutils.core import setup

try:
   import pypandoc
   long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   long_description = ''

setup(name='postgkyl',
      version='0.1.0',
      description='Postrocessing utilities for Gkyl simulation framework',
      long_description=long_description,
      url='https://bitbucket.org/ammarhakim/postgkyl',
      packages=['postgkyl',
                'postgkyl.data',
                'postgkyl.diagnostics',
                'postgkyl.tools'],
     )
