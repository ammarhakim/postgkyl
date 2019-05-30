#!/usr/bin/env python
from glob import glob
import os
import time
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
    click.echo('Postgkyl 1.3 {:s} ({:s})'.format(date, sys.platform))
    click.echo(sys.version)
    click.echo('Copyright 2016-2018 Gkeyll Team')
    click.echo('Gkeyll can be used freely for research at universities,')
    click.echo('national laboratories, and other non-profit institutions.')
    click.echo('There is NO warranty. SPAM!\n')
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

 
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

# # Modifying the docstring for the main file
# fName = os.path.dirname(os.path.realos.path(__file__)) + '/pgkyl.rst'
# with open(fName, 'r') as file:
#     docString = file.read()

# The command line mode entry command
#@click.group(chain=True)
@click.command(cls=AliasedGroup, chain=True,
               context_settings=CONTEXT_SETTINGS)
@click.option('--filename', '-f', multiple=True,
              help="Specify one or more files to work with.")
@click.option('--label', '-l', multiple=True,
              help="Specify a custom label for each dataset.")
@click.option('--savechain', '-s', is_flag=True,
              help="Save command chain for quick repetition.")
@click.option('--savechainas', 
              help="Save command chain for quick repetition with specified name for multiple options")
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
@click.option('--compgrid', is_flag=True,
              help="Disregard the mapped grid information")
@click.pass_context
def cli(ctx, filename, label, savechain, savechainas, stack, verbose,
        c0, c1, c2, c3, c4, c5, comp, compgrid):
    ctx.obj = {}  # The main contex object
    ctx.obj['startTime'] = time.time()  # Timings are written in the verbose mode
    if verbose:
        ctx.obj['verbose'] = True
        # Monty Python references should be a part of Python code
        vlog(ctx, 'This is Postgkyl running in verbose mode!')
        vlog(ctx, 'Spam! Spam! Spam! Spam! Lovely Spam! Lovely Spam!')
        vlog(ctx, 'And now for something completelly different...')
    else:
        ctx.obj['verbose'] = False
    #end
    
    home = os.path.expanduser('~')
    ctx.obj['savechainPath'] = (home + '/.pgkyl/pgkylchain')
    if savechain or savechainas is not None:
        ctx.obj['savechain'] = True
        try:
            os.makedirs(home + "/.pgkyl")
        except FileExistsError:
            # directory already exists
            pass
        #end
        if savechainas is not None:
            ctx.obj['savechainPath'] = (home + '/.pgkyl/' + str(savechainas))
        #end
        fh = open(ctx.obj['savechainPath'], 'w')  # The default chain name
        fh.close()
    else:
        ctx.obj['savechain'] = False
    #end

    ctx.obj['files'] = filename
    numFiles = len(filename)
    ctx.obj['dataSets'] = []
    ctx.obj['labels'] = []
    ctx.obj['setIds'] = []

    ctx.obj['fig'] = ''
    ctx.obj['ax'] = ''

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
        if "*" not in filename[s] and "?" not in filename[s] and "!" not in filename[s]:
            vlog(ctx, "Loading '{:s}\' as data set #{:d}".
                 format(filename[s], cnt))
            try:
                ctx.obj['dataSets'].append(GData(filename[s], comp=comp[s],
                                                 coord0=c0[s], coord1=c1[s],
                                                 coord2=c2[s], coord3=c3[s],
                                                 coord4=c4[s], coord5=c5[s],
                                                 stack=stack,
                                                 compgrid=compgrid))
            except NameError:
                click.echo(click.style("ERROR: File(s) '{:s}' not found or empty".format(filename[s]), fg='red'))
                ctx.exit()
            #end
            ctx.obj['setIds'].append(cnt)
            cnt += 1
        else:  # Postgkyl allows for wild-card loading (requires quotes)
            files = glob(str(filename[s]))
            files = [f for f in files if f.find("restart") < 0]
            def crush(s):                             
                splitted = s.split('_')
                tmp = splitted[-1].split('.')
                splitted[-1] = int(tmp[0])
                splitted.append(tmp[1])
                return tuple(splitted)
            try:
                files = sorted(files, key=crush)
            except Exception:
                click.echo(click.style("WARNING: The loaded files appear to be of different types. Sorting is turned off.", fg='yellow'))
            #end
            
            for fn in files:
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
                #end
            #end
        #end
    #end
    
    # It is possite to run pgkyl without any file (e.e., for getting a
    # holp for a command; however, it will fail if files are specified
    # but not loaded
    if numFiles > 0 and cnt == 0:
        click.echo(click.style("ERROR: No files loaded",fg='red'))
        ctx.exit()
    #end
    ctx.obj['sets'] = range(cnt)

    # Automatically set label from unique parts of the file names
    if cnt > 0:
        nameComps = np.zeros(cnt, np.int)
        names = []
        for sIdx in range(cnt):
            fName = ctx.obj['dataSets'][sIdx].fName
            fName = fName.split('.')[0]
            names.append(fName.split('_'))
            nameComps[sIdx] = len(names[sIdx])
            ctx.obj['labels'].append('')
        #end
        maxComps = np.max(nameComps)
        idxMaxComps = np.argmax(nameComps)
        for i in range(maxComps): 
            unique = True
            compStr = names[idxMaxComps][i]
            for sIdx in range(cnt):
                if i < len(names[sIdx]) and sIdx != idxMaxComps:
                    if names[sIdx][i] == compStr:
                        unique = False
                    #end
                #end
            #end
            if unique:
                for sIdx in range(cnt):
                    if i < len(names[sIdx]):
                        if ctx.obj['labels'][sIdx] == "":
                            ctx.obj['labels'][sIdx] += names[sIdx][i]
                        else:
                            ctx.obj['labels'][sIdx] += '_{:s}'.format(names[sIdx][i])
                        #end
                    #end
                #end
            #end
            # User specified labels
            for sIdx, l in enumerate(label):
                ctx.obj['labels'][sIdx] = l
            #end
        #end
    #end    
#end

@click.command(help='Run the saved command chain')
@click.option('--filename', '-f',
              help="Specify file with stored chain (default 'pgkylchain.dat')")
@click.pass_context
def runchain(ctx, filename):
    if filename is None:
        fn = ctx.obj['savechainPath']
    else:
        home = os.path.expanduser('~')
        fn = home + '/.pgkyl/' + filename
    #end
    if os.path.isfile(fn):
        fh = open(fn, 'r')
        for line in fh.readlines():
            eval('ctx.invoke(cmd.{:s})'.format(line))
        fh.close()
    else:
        raise NameError("File with stored chain ({:s}) does not exist".
                        format(filename))
    #end
#end

@click.command(help='Pop the data stack')
@click.pass_context
def pop(ctx):
    vlog(ctx, 'Poping the stack')
    pushChain(ctx, 'pop')
    for s in ctx.obj['sets']:
        ctx.obj['dataSets'][s].popGrid()
        ctx.obj['dataSets'][s].popValues()
    #end
#end

# Hook the individual commands into pgkyl
cli.add_command(cmd.animate)
cli.add_command(cmd.collect)
cli.add_command(cmd.dataset)
cli.add_command(cmd.euler)
cli.add_command(cmd.ev)
cli.add_command(cmd.fft)
cli.add_command(cmd.growth)
cli.add_command(cmd.info)
cli.add_command(cmd.extractinput)
cli.add_command(cmd.integrate)
cli.add_command(cmd.interpolate)
cli.add_command(cmd.plot)
cli.add_command(cmd.pr)
cli.add_command(cmd.recovery)
cli.add_command(cmd.select)
cli.add_command(cmd.tenmoment)
cli.add_command(cmd.write)
cli.add_command(pop)
cli.add_command(runchain)

cli.add_command(cmd.agyro)
cli.add_command(cmd.temp.norm)

#cli.add_command(cmd.cglpressure.cglpressure)

#cli.add_command(cmd.transform.curl)
#cli.add_command(cmd.transform.div)
#cli.add_command(cmd.transform.grad)
#cli.add_command(cmd.transform.mask)

if __name__ == '__main__':
    cli()


