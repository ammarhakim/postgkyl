import click
import numpy as np

import postgkyl.data.select 
from postgkyl.commands.util import vlog, pushChain

@click.command()
@click.option('--z0',  default=None,
              help="Indices for 0th coord (either int, float, or slice)")
@click.option('--z1', default=None,
              help="Indices for 1st coord (either int, float, or slice)")
@click.option('--z2', default=None,
              help="Indices for 2nd coord (either int, float, or slice)")
@click.option('--z3', default=None,
              help="Indices for 3rd coord (either int, float, or slice)")
@click.option('--z4', default=None,
              help="Indices for 4th coord (either int, float, or slice)")
@click.option('--z5', default=None,
              help="Indices for 5th coord (either int, float, or slice)")
@click.option('--comp', '-c', default=None,
              help="Indices for components (either int, slice, or coma-separated)")
@click.option('--tag', '-t',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.pass_context
def select(ctx, **kwargs):
    r"""Subselect data from the active dataset(s). This command allows, for
    example, to choose a specific component of a multi-component
    dataset, select a index or coordinate range. Index ranges can also
    be specified using python slice notation (start:end:stride).

    """
    vlog(ctx, 'Starting select')
    pushChain(ctx, 'select', **kwargs)
    for dat in ctx.obj['data'].iterator(kwargs['tag']):
        postgkyl.data.select(dat, stack=True,
                             z0=kwargs['z0'], z1=kwargs['z1'],
                             z2=kwargs['z2'], z3=kwargs['z3'],
                             z4=kwargs['z4'], z5=kwargs['z5'],
                             comp=kwargs['comp'])
    #end
    vlog(ctx, 'Finishing select')
#end
