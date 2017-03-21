#!/usr/bin/env python

from distutils.core import setup

setup(name='postgkyl',
      version='0.1.0',
      description='Gkeyll (and Hyde) Postprocessing Utilities',
      url='https://bitbucket.org/ammarhakim/postgkyl',
      packages=['postgkyl',
                'postgkyl.data',
                'postgkyl.diagnostics',
                'postgkyl.tools'],
     )
