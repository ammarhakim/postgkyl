#!/bin/bash

# Modify the build number to the day of the last change
./setbuildnumber.py

conda config --set anaconda_upload yes

# Linux builds
conda build . --python=3.6 --numpy 1.21.5 --channel conda-forge
conda build . --python=3.7 --numpy 1.21.5 --channel conda-forge
conda build . --python=3.8 --numpy 1.21.5 --channel conda-forge
conda build . --python=3.9 --numpy 1.21.5 --channel conda-forge

# Convert
OS=$(uname)
PKG36=$(conda build . --python=3.6 --output)
PKG37=$(conda build . --python=3.7 --output)
PKG38=$(conda build . --python=3.8 --output)
PKG39=$(conda build . --python=3.9 --output)
if [ $OS="Linux" ]
then 
    conda convert --platform osx-64 $PKG36
    conda convert --platform osx-64 $PKG37
    conda convert --platform osx-64 $PKG38
    conda convert --platform osx-64 $PKG39
    cd osx-64
    anaconda upload * --force
    cd ..
    rm -rf osx-64
else
    conda convert --platform linux-64 $PKG36
    conda convert --platform linux-64 $PKG37
    conda convert --platform linux-64 $PKG38
    conda convert --platform linux-64 $PKG39
    cd linux-64
    anaconda upload * --force
    cd ..
    rm -rf linux-64
fi
conda build purge
