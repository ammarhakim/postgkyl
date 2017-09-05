#!/usr/bin/env python
import click
import numpy
import os
from glob import glob
from time import time

import postgkyl.commands as cmd
from postgkyl.tools.stack import pushStack, peakStack, popStack
from postgkyl.tools.stack import loadFrame, loadHist
from postgkyl.commands.output import vlog

@click.group(chain=True)
@click.option('-v', '--verbose', is_flag=True,
              help='Turn on verbosity')
@click.option('--filename', '-f', multiple=True,
              help='Specify one or more file(s) to work with.')
@click.option('--histname', '-h', multiple=True,
              help='Specify one or more history file(s) to work with.')
@click.pass_context
def cli(ctx, filename, histname, verbose):
    ctx.obj = {}

    ctx.obj['startTime'] = time()
    if verbose:
        ctx.obj['verbose'] = True
        vlog(ctx, 'This is postgkyl running in verbose mode!')
        vlog(ctx, 'Spam! Spam! Spam! Spam! Lovely Spam! Lovely Spam!')
        vlog(ctx, 'And now for something completelly different...')
    else:
        ctx.obj['verbose'] = False

    ctx.obj['files'] = filename
    numFileSets = len(filename)
    numHistSets = len(histname)
    ctx.obj['type'] = []
    ctx.obj['setIds'] = []

    ctx.obj['data'] = []
    ctx.obj['labels'] = []

    ctx.obj['coords'] = []
    ctx.obj['values'] = []

    cnt = 0
    for s in range(numFileSets):
        if filename[s][-2:] == 'h5' or filename[s][-2:] == 'bp':
            files = glob(str(filename[s]))
            for i in range(len(files)):
                vlog(ctx, 'Loading frame \'{:s}\' as data set #{:d}'.format(files[i], cnt))
                loadFrame(ctx, cnt, files[i])
                cnt += 1
        else:
            if ctx.obj['verbose']:
                click.echo(click.style('Loading history \'{:s}\' as data set #{:d}'.format(filename[s], cnt), fg='green'))
            loadHist(ctx, cnt, str(filename[s]))
            cnt += 1            

    for s in range(numHistSets):
        if histname[s][-2:] == 'h5' or histname[s][-2:] == 'bp':
            if ctx.obj['verbose']:
                click.echo(click.style('Loading history \'{:s}\' as data set #{:d}'.format(histname[s], cnt), fg='green'))
            loadHist(ctx, cnt, str(histname[s]))
            cnt += 1
            
    ctx.obj['numSets'] = cnt
    ctx.obj['sets'] = range(cnt)

    ctx.obj['hold'] = 'off'
    ctx.obj['fig'] = ''
    ctx.obj['ax'] = ''

    dirPath = os.path.dirname(os.path.realpath(__file__))
    if os.path.isfile(dirPath+'/postgkyl.mplstyle'): 
        ctx.obj['mplstyle'] = dirPath + '/postgkyl.mplstyle'
    else:
        ctx.obj['mplstyle']  = dirPath + '/../../../../data/postgkyl.mplstyle'

cli.add_command(cmd.cglpressure.cglpressure)
cli.add_command(cmd.diagnostics.growth)
cli.add_command(cmd.euler.euler)
cli.add_command(cmd.output.hold)
cli.add_command(cmd.output.info)
cli.add_command(cmd.output.plot)
cli.add_command(cmd.output.write)
cli.add_command(cmd.select.collect)
cli.add_command(cmd.select.comp)
cli.add_command(cmd.select.dataset)
cli.add_command(cmd.select.fix)
cli.add_command(cmd.select.pop)
cli.add_command(cmd.tenmoment.tenmoment)
cli.add_command(cmd.transform.abs)
cli.add_command(cmd.transform.curl)
cli.add_command(cmd.transform.div)
cli.add_command(cmd.transform.fft)
cli.add_command(cmd.transform.grad)
cli.add_command(cmd.transform.integrate)
cli.add_command(cmd.transform.interpolate)
cli.add_command(cmd.transform.log)
cli.add_command(cmd.transform.mask)
cli.add_command(cmd.transform.mult)
cli.add_command(cmd.transform.norm)
cli.add_command(cmd.transform.pow)
cli.add_command(cmd.transform.transpose)

if __name__ == '__main__':
    cli()

