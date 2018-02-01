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
