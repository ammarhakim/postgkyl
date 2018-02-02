from time import time
from os.path import isfile
import base64
import sys

import click

def vlog(ctx, message):
    if ctx.obj['verbose']:
        elapsedTime = time() - ctx.obj['startTime']
        click.echo(click.style('[{:f}] {:s}'.format(elapsedTime, message),
                               fg='green'))

def pushChain(ctx, command, **kwargs):
    if ctx.obj['savechain'] == True:
        fh = open('pgkylchain.dat', 'a')
        s = '{:s}'.format(command)

        for key, value in kwargs.items():
            if sys.version_info[0] == 2:
                if type(value) == unicode:
                    value = str(value)
            if type(value) == str:
                value = "'" + value + "'"
            s = s + ', {:s}={}'.format(key, value)
        if sys.version_info[0] == 3:
            fh.write(base64.b64encode(s.encode()).decode())
        else:
            fh.write(base64.b64encode(s.encode()))
        fh.write('\n')
        fh.close()

