#!/bin/bash

# Modify the build number to the day of the last change
export CUSTOM_BUILD_NUMBER=$(date +'%Y%m%d')
conda config --set anaconda_upload yes

# Linux builds
conda build . --python=3.11 --channel conda-forge

# Convert
OS=$(uname)
PKG311=$(conda build . --python=3.11 --output)
if [ $OS="Linux" ]
then
    conda convert --platform osx-64 $PKG311
    cd osx-64
    anaconda upload * --force
    cd ..
    rm -rf osx-64
else
    conda convert --platform linux-64 $PKG311
    cd linux-64
    anaconda upload * --force
    cd ..
    rm -rf linux-64
fi
conda build purge
