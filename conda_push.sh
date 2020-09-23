#!/bin/bash
./setbuildnumber.py
conda config --set anaconda_upload yes
conda build . --python=3.6
conda build . --python=3.7
conda build . --python=3.8
