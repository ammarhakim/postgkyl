#!/bin/bash
conda config --set anaconda_upload yes
conda build . --python=3.6
conda build . --python=2.7
