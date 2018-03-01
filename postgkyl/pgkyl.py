#!/usr/bin/env python
from glob import glob
from os.path import isfile
from time import time
import sys

import click
import numpy as np

from postgkyl.commands.util import vlog, pushChain
from postgkyl.data import GData
import postgkyl.commands as cmd

# Version print helper
def _printVersion(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo('Postgkyl 1.0 2018-02 ({:s})'.format(sys.platform))
    click.echo(sys.version)
    click.echo('Copyright 2016-2018 Gkyl Team')
    click.echo('Gkyl can be used freely for research at universities,')
    click.echo('national laboratories and other non-profit institutions.')
    click.echo('There is NO warranty.\n')
    ctx.exit()

# Helper for expanding the partial load indices for easier looping
def _expandPartialLoadIdx(numFiles, idx):
    if not idx:
        idxOut = [None for _ in range(numFiles)]
    elif len(idx) == 1:
        idxOut = [idx[0] for _ in range(numFiles)]
    elif len(idx) > 1 and len(idx) != numFiles:
        raise IndexError("Partial load indices mismatch")
    else:
        idxOut = idx
    return idxOut

class AliasedGroup(click.Group):

    def get_command(self, ctx, cmd_name):
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        matches = [x for x in self.list_commands(ctx)
                   if x.startswith(cmd_name)]
        if not matches:
            return None
        elif len(matches) == 1:
            return click.Group.get_command(self, ctx, matches[0])
        ctx.fail('Too many matches: %s' % ', '.join(sorted(matches)))
 
# The command line mode entry command
#@click.group(chain=True)
@click.command(cls=AliasedGroup, chain=True)
@click.option('--filename', '-f', multiple=True,
              help="Specify one or more files to work with.")
@click.option('--savechain', '-s', is_flag=True,
              help="Save command chain for quick repetition.")
@click.option('--stack/--no-stack', default=False,
              help="Turn the Postgkyl stack capabilities ON/OFF")
@click.option('--verbose', '-v', is_flag=True,
              help="Turn on verbosity.")
@click.option('--version', is_flag=True, callback=_printVersion,
              expose_value=False, is_eager=True,
              help="Print the version information.")
@click.option('--c0', multiple=True,
              help="Partial file load: 0th coord (either int or slice)")
@click.option('--c1', multiple=True,
              help="Partial file load: 1st coord (either int or slice)")
@click.option('--c2', multiple=True,
              help="Partial file load: 2nd coord (either int or slice)")
@click.option('--c3', multiple=True,
              help="Partial file load: 3rd coord (either int or slice)")
@click.option('--c4', multiple=True,
              help="Partial file load: 4th coord (either int or slice)")
@click.option('--c5', multiple=True,
              help="Partial file load: 5th coord (either int or slice)")
@click.option('--comp', '-c', multiple=True,
              help="Partial file load: comps (either int or slice)")
@click.pass_context
def cli(ctx, filename, savechain, stack, verbose,
        c0, c1, c2, c3, c4, c5, comp):
    ctx.obj = {}  # The main contex object
    ctx.obj['startTime'] = time()  # Timings are written in the verbose mode
    if verbose:
        ctx.obj['verbose'] = True
        # Monty Python references should be a part of Python code
        vlog(ctx, 'This is Postgkyl running in verbose mode!')
        vlog(ctx, 'Spam! Spam! Spam! Spam! Lovely Spam! Lovely Spam!')
        vlog(ctx, 'And now for something completelly different...')
    else:
        ctx.obj['verbose'] = False

    if savechain:
        ctx.obj['savechain'] = True
        fh = open('.pgkylchain', 'w')  # The default chain name
        fh.close()
    else:
        ctx.obj['savechain'] = False

    ctx.obj['files'] = filename
    numFiles = len(filename)
    ctx.obj['dataSets'] = []
    ctx.obj['setIds'] = []

    # Expand indices for easy looping
    c0 = _expandPartialLoadIdx(numFiles, c0)
    c1 = _expandPartialLoadIdx(numFiles, c1)
    c2 = _expandPartialLoadIdx(numFiles, c2)
    c3 = _expandPartialLoadIdx(numFiles, c3)
    c4 = _expandPartialLoadIdx(numFiles, c4)
    c5 = _expandPartialLoadIdx(numFiles, c5)
    comp = _expandPartialLoadIdx(numFiles, comp)

    cnt = 0 # Counter for number of loaded files
    for s in range(numFiles):
        if "*" not in filename[s] and "?" not in filename[s]:
            vlog(ctx, "Loading '{:s}\' as data set #{:d}".
                 format(filename[s], cnt))
            ctx.obj['dataSets'].append(GData(filename[s], comp=comp[s],
                                             coord0=c0[s], coord1=c1[s],
                                             coord2=c2[s], coord3=c3[s],
                                             coord4=c4[s], coord5=c5[s],
                                             stack=stack))
            ctx.obj['setIds'].append(cnt)
            cnt += 1
        else:  # Postgkyl allows for wild-card loading (requires quotes)
            files = glob(str(filename[s]))
            for fn in sorted(files):
                try:
                    vlog(ctx, "Loading '{:s}\' as data set #{:d}".
                         format(fn, cnt))
                    ctx.obj['dataSets'].append(GData(fn, comp=comp[s],
                                                     coord0=c0[s],
                                                     coord1=c1[s],
                                                     coord2=c2[s],
                                                     coord3=c3[s],
                                                     coord4=c4[s],
                                                     coord5=c5[s],
                                                     stack=stack))
                    ctx.obj['setIds'].append(cnt)
                    cnt += 1
                except:
                    pass
    # It is possite to run pgkyl without any file (e.e., for getting a
    # holp for a command; however, it will fail if files are specified
    # but not loaded
    if numFiles > 0 and cnt == 0:
        raise NameError("no files loaded")
    ctx.obj['sets'] = range(cnt)

    ctx.obj['fig'] = ''
    ctx.obj['ax'] = ''

@click.command(help='Run the saved command chain')
@click.option('--filename', '-f', default='.pgkylchain',
              help="Specify file with stored chain (default 'pgkylchain.dat')")
@click.pass_context
def runchain(ctx, filename):
    if isfile(filename):
        fh = open(filename, 'r')
        for line in fh.readlines():
            eval('ctx.invoke(cmd.{:s})'.format(line))
        fh.close()
    else:
        raise NameError("File with stored chain ({:s}) does not exist".
                        format(filename))

@click.command(help='Pop the data stack')
@click.pass_context
def pop(ctx):
    vlog(ctx, 'Poping the stack')
    pushChain(ctx, 'pop')
    for s in ctx.obj['sets']:
        ctx.obj['dataSets'][s].popGrid()
        ctx.obj['dataSets'][s].popValues()

# Hook the individual commands into pgkyl
cli.add_command(cmd.dg.interpolate)
cli.add_command(cmd.collect.collect)
cli.add_command(cmd.info.info)
cli.add_command(cmd.plot.plot)
cli.add_command(cmd.select.dataset)
cli.add_command(cmd.select.select)
cli.add_command(cmd.calculus.integrate)
cli.add_command(runchain)
cli.add_command(pop)

#cli.add_command(cmd.agyro.agyro)
#cli.add_command(cmd.cglpressure.cglpressure)
#cli.add_command(cmd.diagnostics.growth)
#cli.add_command(cmd.euler.euler)
#cli.add_command(cmd.output.hold)
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


