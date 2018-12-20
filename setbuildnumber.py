#!/usr/bin/env python
from os import path
from glob import glob
import time

fls = glob(path.dirname(path.realpath(__file__)) + "/*/*/*.py")
latest = 0.0
for f in fls:
    if latest < path.getmtime(f):
        latest = path.getmtime(f)
        struct = time.gmtime(latest)
        date = "{:d}{:02d}{:02d}".format(struct.tm_year,
                                         struct.tm_mon,
                                         struct.tm_mday)

with open(path.dirname(path.realpath(__file__)) + "/meta.yaml", "r") as fh:
    fl = fh.readlines()
    for i, l in enumerate(fl): 
        if l.find('number') > 0: 
            fl[i] = fl[i][:10] + date + '\n'

fh = open(path.dirname(path.realpath(__file__)) + "/meta.yaml", "w")
for l in fl:
    fh.write(l)
fh.close()
