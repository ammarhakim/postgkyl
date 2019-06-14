import click
import numpy as np

import postgkyl.data.select 
from postgkyl.commands.util import vlog, pushChain

@click.command(help='Subselect data set(s)')
@click.option('--c0', default=None,
              help="Indices for 0th coord (either int, float, or slice)")
@click.option('--c1', default=None,
              help="Indices for 1st coord (either int, float, or slice)")
@click.option('--c2', default=None,
              help="Indices for 2nd coord (either int, float, or slice)")
@click.option('--c3', default=None,
              help="Indices for 3rd coord (either int, float, or slice)")
@click.option('--c4', default=None,
              help="Indices for 4th coord (either int, float, or slice)")
@click.option('--c5',  default=None,
              help="Indices for 5th coord (either int, float, or slice)")
@click.option('--comp', '-c', default=None,
              help="Indices for components (either int, slice, or coma-separated)")
@click.pass_context
def select(ctx, **kwargs):
    vlog(ctx, 'Starting select')
    pushChain(ctx, 'select', **kwargs)
    for s in ctx.obj['sets']:
        postgkyl.data.select(ctx.obj['dataSets'][s], stack=True,
                             coord0=kwargs['c0'], coord1=kwargs['c1'],
                             coord2=kwargs['c2'], coord3=kwargs['c3'],
                             coord4=kwargs['c4'], coord5=kwargs['c5'],
                             comp=kwargs['comp'])
    #end
    vlog(ctx, 'Finishing select')
#end
