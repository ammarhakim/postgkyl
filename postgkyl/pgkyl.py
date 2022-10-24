#!/usr/bin/env python3
from glob import glob
import os
import time
import sys
from os import path

import click
import numpy as np

from postgkyl.commands import DataSpace
from postgkyl.commands.util import vlog, pushChain
from postgkyl.data import Data
import postgkyl.commands as cmd

# Version print helper
def _printVersion(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    #end        
    fls = glob(os.path.dirname(os.path.realpath(__file__)) + "/*/*.py")
    latest = 0.0
    for f in fls:
        if latest < os.path.getmtime(f):
            latest = os.path.getmtime(f)
            struct = time.gmtime(latest)
            date = "{:d}-{:02d}-{:02d}".format(struct.tm_year,
                                               struct.tm_mon,
                                               struct.tm_mday)
        #end
    #end
    click.echo('Postgkyl 1.6.8 {:s} ({:s})'.format(date, sys.platform))
    click.echo('Python version: {:s}'.format(sys.version))
    click.echo('Copyright 2016-2021 Gkeyll Team')
    click.echo('Postgkyl can be used freely for research at universities,')
    click.echo('national laboratories, and other non-profit institutions.')
    click.echo('There is NO warranty.\n')
    click.echo('Spam, egg, sausage, and spam.')
    ctx.exit()
#end

# Custom click class that allows to
#   a) use shortened versions of command names
#   b) use a file name as a command
class PgkylCommandGroup(click.Group):
    def get_command(self, ctx, cmdName):
        # cmdName is a full name of a pgkyl command
        rv = click.Group.get_command(self, ctx, cmdName)
        if rv is not None:
            return rv
        #end

        # cmdName is an abreviation of a pgkyl command
        matches = [x for x in self.list_commands(ctx)
                   if x.startswith(cmdName)]
        if matches and len(matches) == 1:
            return click.Group.get_command(self, ctx, matches[0])
        elif matches:
            ctx.fail("Too many matches for '{:s}': {:s}".format(cmdName, ', '.join(sorted(matches))))
        #end

        # cmdName is a data set
        if glob(cmdName):
            ctx.obj['inDataStrings'].append(cmdName)
            return click.Group.get_command(self, ctx, 'load')
        #end
        
        ctx.fail("'{:s}' does not match either command name nor a data file".format(cmdName))
    #end
#end

# The command line mode entry command
@click.command(cls=PgkylCommandGroup, chain=True,
               context_settings=dict(help_option_names=['-h', '--help']))
@click.option('--savechain', '-s', is_flag=True,
              help="Save command chain for quick repetition.")
@click.option('--savechainas', 
              help="Save command chain for quick repetition with specified name for multiple options")
@click.option('--verbose', '-v', is_flag=True,
              help="Turn on verbosity.")
@click.option('--version', is_flag=True, callback=_printVersion,
              expose_value=False, is_eager=True,
              help="Print the version information.")
@click.option('--z0', help="Partial file load: 0th coord (either int or slice)")
@click.option('--z1', help="Partial file load: 1st coord (either int or slice)")
@click.option('--z2', help="Partial file load: 2nd coord (either int or slice)")
@click.option('--z3', help="Partial file load: 3rd coord (either int or slice)")
@click.option('--z4', help="Partial file load: 4th coord (either int or slice)")
@click.option('--z5', help="Partial file load: 5th coord (either int or slice)")
@click.option('--component', '-c',
              help="Partial file load: comps (either int or slice)")
@click.option('--compgrid', is_flag=True,
              help="Disregard the mapped grid information")
@click.option('--varname', '-d', multiple=True,
              help="Specify the Adios variable name (default is 'CartGridField')")
@click.option('--c2p', 
              help="Specify the file name containing c2p mapped coordinates")
@click.pass_context
def cli(ctx, **kwargs):
    """Postprocessing and plotting tool for Gkeyll 
    data. Datasets can be loaded, processed and plotted using a
    command chaining mechanism. For full documentation see the Gkeyll
    documentation webpages. Help for individual commands can be
    obtained using the --help option for that command.
    """
    ctx.obj = {}  # The main contex object
    ctx.obj['startTime'] = time.time()  # Timings are written in the verbose mode
    if kwargs['verbose']:
        ctx.obj['verbose'] = True
        # Monty Python references should be a part of any Python code
        vlog(ctx, 'This is Postgkyl running in verbose mode!')
        vlog(ctx, 'Spam! Spam! Spam! Spam! Lovely Spam! Lovely Spam!')
        vlog(ctx, 'And now for something completelly different...')
    else:
        ctx.obj['verbose'] = False
    #end
    
    home = os.path.expanduser('~')
    ctx.obj['savechainPath'] = (home + '/.pgkyl/pgkylchain')
    if kwargs['savechain'] or kwargs['savechainas'] is not None:
        ctx.obj['savechain'] = True
        try:
            os.makedirs(home + "/.pgkyl")
        except FileExistsError:
            # directory already exists
            pass
        #end
        if kwargs['savechainas'] is not None:
            ctx.obj['savechainPath'] = (home + '/.pgkyl/' + str(kwargs['savechainas']))
        #end
        fh = open(ctx.obj['savechainPath'], 'w')  # The default chain name
        fh.close()
    else:
        ctx.obj['savechain'] = False
    #end

    ctx.obj['inDataStrings'] = []
    ctx.obj['inDataStringsLoaded'] = 0
    
    ctx.obj['data'] = DataSpace()

    ctx.obj['fig'] = ''
    ctx.obj['ax'] = ''

    ctx.obj['compgrid'] = kwargs['compgrid']
    ctx.obj['globalVarNames'] = kwargs['varname']    
    ctx.obj['globalCuts'] = (kwargs['z0'], kwargs['z1'],
                             kwargs['z2'], kwargs['z3'],
                             kwargs['z4'], kwargs['z5'],
                             kwargs['component'])
    ctx.obj['global_c2p'] = kwargs['c2p']
#end

# Hook the individual commands into pgkyl
cli.add_command(cmd.activate)
cli.add_command(cmd.animate)
cli.add_command(cmd.collect)
cli.add_command(cmd.copystyle)
cli.add_command(cmd.current)
cli.add_command(cmd.deactivate)
cli.add_command(cmd.differentiate)
cli.add_command(cmd.energetics)
cli.add_command(cmd.euler)
cli.add_command(cmd.mhd)
cli.add_command(cmd.ev)
cli.add_command(cmd.extractinput)
cli.add_command(cmd.fft)
cli.add_command(cmd.growth)
cli.add_command(cmd.info)
cli.add_command(cmd.integrate)
cli.add_command(cmd.interpolate)
cli.add_command(cmd.listoutputs)
cli.add_command(cmd.load)
cli.add_command(cmd.magsq)
cli.add_command(cmd.mask)
cli.add_command(cmd.plot)
#cli.add_command(cmd.pop)
cli.add_command(cmd.pr)
cli.add_command(cmd.recovery)
cli.add_command(cmd.relchange)
cli.add_command(cmd.runchain)
cli.add_command(cmd.select)
cli.add_command(cmd.tenmoment)
cli.add_command(cmd.trajectory)
cli.add_command(cmd.val2coord)
cli.add_command(cmd.velocity)
cli.add_command(cmd.write)

#cli.add_command(cmd.agyro)
#cli.add_command(cmd.temp.norm)

if __name__ == '__main__':
    ctx = []
    cli(ctx)


