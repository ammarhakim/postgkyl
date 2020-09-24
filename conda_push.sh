#!/bin/bash

# Modify the build number to the day of the last change
./setbuildnumber.py

conda config --set anaconda_upload yes

# Linux builds
conda build . --python=3.6
conda build . --python=3.7
conda build . --python=3.8

# Convert
set OS (uname)
set pkg36 (conda build . --python=3.6 --output)
set pkg37 (conda build . --python=3.7 --output)
set pkg38 (conda build . --python=3.8 --output)
if [ $OS="Linux" ]
then
    conda convert --platform osx-64 pkg36
    conda convert --platform osx-64 pkg37
    conda convert --platform osx-64 pkg38
    cd osx-64
    anaconda upload pkg36
    anaconda upload pkg37
    anaconda upload pkg38
    cd ..
    rm -rf osx-64
else
    conda convert --platform linux-64 pkg36
    conda convert --platform linux-64 pkg37
    conda convert --platform linux-64 pkg38
    cd linux-64
    anaconda upload pkg36
    anaconda upload pkg37
    anaconda upload pkg38
    cd ..
    rm -rf linux-64
fi
conda build purge
