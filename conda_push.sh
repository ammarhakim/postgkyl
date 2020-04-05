#!/bin/bash
conda config --set anaconda_upload yes
conda build .
conda build . --python=3.6

