#!/usr/bin/env python
import base64
import sys
from glob import glob
from time import time
from os.path import isfile

import click
import numpy as np

from postgkyl.data import GData
import postgkyl.commands as cmd
from postgkyl.commands.util import vlog

@click.group(chain=True)
@click.option('-v', '--verbose', is_flag=True,
              help='Turn on verbosity')
@click.option('--filename', '-f', multiple=True,
              help='Specify one or more file(s) to work with.')
@click.option('--savechain', '-c', is_flag=True,
              help='Save command chain for quick repetition')
@click.pass_context
def cli(ctx, filename, verbose, savechain):
    ctx.obj = {}

    ctx.obj['startTime'] = time()
    if verbose:
        ctx.obj['verbose'] = True
        vlog(ctx, 'This is Postgkyl running in verbose mode!')
        vlog(ctx, 'Spam! Spam! Spam! Spam! Lovely Spam! Lovely Spam!')
        vlog(ctx, 'And now for something completelly different...')
    else:
        ctx.obj['verbose'] = False

    if savechain:
        ctx.obj['savechain'] = True
        fh = open('pgkylchain.dat', 'w')
        fh.close()
    else:
        ctx.obj['savechain'] = False

    ctx.obj['files'] = filename
    numFiles = len(filename)
    ctx.obj['dataSets'] = []
    ctx.obj['setIds'] = []

    cnt = 0
    for s in range(numFiles):
        if "*" not in filename[s]:
            vlog(ctx, "Loading '{:s}\' as data set #{:d}".
                 format(filename[s], cnt))
            ctx.obj['dataSets'].append(GData(filename[s]))
            ctx.obj['setIds'].append(cnt)
            cnt += 1
        else:
            files = glob(str(filename[s]))
            for fn in files:
                try:
                    vlog(ctx, "Loading '{:s}\' as data set #{:d}".
                         format(fn, cnt))
                    ctx.obj['dataSets'].append(GData(fn))
                    ctx.obj['setIds'].append(cnt)
                    cnt += 1
                except:
                    pass

    if numFiles > 0 and cnt == 0:
        raise NameError("no files loaded")
    ctx.obj['numSets'] = cnt
    ctx.obj['sets'] = range(cnt)

    ctx.obj['hold'] = 'off'
    ctx.obj['fig'] = ''
    ctx.obj['ax'] = ''

@click.command(help='Run the saved command chain')
@click.pass_context
def rc(ctx):
    if isfile('pgkylchain.dat'):
        fh = open('pgkylchain.dat', 'r')
        for line in fh.readlines():
            if sys.version_info[0] == 3:
                s = base64.b64decode(line).decode()
            else:
                s = base64.b64decode(line)
            eval('ctx.invoke(cmd.{:s})'.format(s))
        fh.close()
    else:
        click.echo("WARNING: 'pgkylchain.dat' does not exist; "
                   "command chain needs to be saved first with "
                   "the pgkyl flag -c")


cli.add_command(rc)
cli.add_command(cmd.info.info)
cli.add_command(cmd.dg.interpolate)
cli.add_command(cmd.plot.plot)
cli.add_command(cmd.select.fix)
cli.add_command(cmd.select.comp)

#cli.add_command(cmd.agyro.agyro)
#cli.add_command(cmd.cglpressure.cglpressure)
#cli.add_command(cmd.diagnostics.growth)
#cli.add_command(cmd.euler.euler)
#cli.add_command(cmd.output.hold)
#cli.add_command(cmd.output.plot)
#cli.add_command(cmd.output.write)
#cli.add_command(cmd.select.collect)
#cli.add_command(cmd.select.comp)
#cli.add_command(cmd.select.dataset)
#cli.add_command(cmd.select.fix)
#cli.add_command(cmd.select.pop)
#cli.add_command(cmd.tenmoment.tenmoment)
#cli.add_command(cmd.transform.abs)
#cli.add_command(cmd.transform.curl)
#cli.add_command(cmd.transform.div)
#cli.add_command(cmd.transform.fft)
#cli.add_command(cmd.transform.grad)
#cli.add_command(cmd.transform.integrate)
#cli.add_command(cmd.transform.interpolate)
#cli.add_command(cmd.transform.log)
#cli.add_command(cmd.transform.mask)
#cli.add_command(cmd.transform.mult)
#cli.add_command(cmd.transform.norm)
#cli.add_command(cmd.transform.pow)
#cli.add_command(cmd.transform.transpose)

if __name__ == '__main__':
    cli()


