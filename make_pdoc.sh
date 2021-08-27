#!/bin/bash
pdoc postgkyl -t pdoc_template/ -o pdoc --docformat google
cp -r pdoc/postgkyl ../gkyldoc/source/postgkyl/_static
