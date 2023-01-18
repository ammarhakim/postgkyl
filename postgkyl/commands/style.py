import click
import shutil
import os.path

from postgkyl.commands.util import load_style, verb_print

@click.command(help='Probe and control the plotting style.')
@click.option('--print', '-p', is_flag=True,
              help="Prints the current rcParams.")
@click.option('--set', '-s', multiple=True,
              help="Sets rcParam(s).")
@click.option('--file', '-f',
              help="Sets Maplotlib rcParams style file.")
@click.pass_context
def style(ctx, **kwargs):
  verb_print(ctx, 'Starting \'style\' command')

  if kwargs['print']:
    for key in ctx.obj['rcParams']:
      print('{:s} : {}'.format(key, ctx.obj['rcParams'][key]))
    #end
  #end

  for param in kwargs['set']:
    param_split = param.split(':')
    key = param_split[0].strip()
    value = param[len(param_split[0])+1:].strip()
    ctx.obj['rcParams'][key] = value
  #end
  
  if  kwargs['file']:
    load_style(ctx, kwargs['file'])
  #end

  verb_print(ctx, 'Finishing \'style\' command')
#end
