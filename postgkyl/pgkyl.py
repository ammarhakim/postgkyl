#!/usr/bin/env python
import click
import numpy
import os
from glob import glob

import postgkyl.commands as cmd
from postgkyl.tools.stack import pushStack, peakStack, popStack
from postgkyl.tools.stack import loadFrame, loadHist

@click.group(chain=True)
@click.option('--filename', '-f', multiple=True,
              help='Specify one or more file(s) to work with.')
@click.pass_context
def cli(ctx, filename):
    ctx.obj = {}

    ctx.obj['files'] = filename
    numSets = len(filename)
    ctx.obj['type'] = []
    ctx.obj['setIds'] = []

    ctx.obj['data'] = []
    ctx.obj['labels'] = []

    ctx.obj['coords'] = []
    ctx.obj['values'] = []

    cnt = 0
    for s in range(numSets):
        if filename[s][-2:] == 'h5' or filename[s][-2:] == 'bp':
            files = glob(str(filename[s]))
            for i in range(len(files)):
                loadFrame(ctx, cnt, files[i])
                cnt += 1
        else:
            loadHist(ctx, cnt, str(filename[s]))
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

cli.add_command(cmd.diagnostics.growth)
cli.add_command(cmd.euler.euler)
cli.add_command(cmd.output.hold)
cli.add_command(cmd.output.info)
cli.add_command(cmd.output.plot)
cli.add_command(cmd.output.write)
cli.add_command(cmd.select.comp)
cli.add_command(cmd.select.dataset)
cli.add_command(cmd.select.fix)
cli.add_command(cmd.select.pop)
cli.add_command(cmd.select.collect)
cli.add_command(cmd.transform.log)
cli.add_command(cmd.transform.abs)
cli.add_command(cmd.transform.curl)
cli.add_command(cmd.transform.div)
cli.add_command(cmd.transform.grad)
cli.add_command(cmd.transform.integrate)
cli.add_command(cmd.transform.mask)
cli.add_command(cmd.transform.mult)
cli.add_command(cmd.transform.pow)
cli.add_command(cmd.transform.fft)
cli.add_command(cmd.transform.norm)
cli.add_command(cmd.transform.project)
cli.add_command(cmd.transform.transpose)

if __name__ == '__main__':
    cli()

