from time import time
import sys

from cycler import cycler
import click

def verb_print(ctx, message):
  if ctx.obj['verbose']:
    elapsedTime = time() - ctx.obj['startTime']
    click.echo(click.style('[{:f}] {:s}'.format(elapsedTime, message),
                           fg='green'))
  #end
#end

def load_style(ctx, fn):
  fh = open(fn, 'r')
  for line in fh.readlines():
    key = line.split(':')[0]
    key_len = int(len(key))
    key = key.strip()
    value = line[(key_len+1):].strip()
    if value[:6] == 'cycler':
      arg = eval(value[16:-1])
      value = cycler(color=arg)
    #end
    ctx.obj['rcParams'][key] = value
  #end
  fh.close()
#end
