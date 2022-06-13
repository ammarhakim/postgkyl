import click

from postgkyl.commands.util import vlog, pushChain
from postgkyl.data import Data
import postgkyl.diagnostics as diag

@click.command()
@click.option('--array', '-a',
              default='array', show_default=True,
              help="Tag for array to be rotated")
@click.option('--rotator', '-r',
              default='rotator', show_default=True,
              help="Tag for rotator (data used for the rotation)")
@click.option('--tag', '-t',
              default='rotarraypar', show_default=True,
              help='Tag for the resulting rotated array parallel to rotator')
@click.option('--label', '-l',
              default='rotarraypar', show_default=True,
              help="Custom label for the result")
@click.pass_context
def parrotate(ctx, **kwargs):
    """Rotate an array parallel to the unit vectors of a second array.
    For two arrays u and v, where v is the rotator, operation is (u dot v_hat) v_hat.
    Note that for a three-component field, the output is a new vector
    whose components are (u_{v_x}, u_{v_y}, u_{v_z}), i.e.,
    the x, y, and z components of the vector u parallel to v. 
    """
    vlog(ctx, 'Starting rotation parallel to rotator array')
    pushChain(ctx, 'rotarraypar', **kwargs)
    
    data = ctx.obj['data'] # shortcut
    
    for a, rot in zip(data.iterator(kwargs['array']),
                      data.iterator(kwargs['rotator'])):
        grid, outrot = diag.parrotate(a, rot)
        # Create new GData structure with appropriate outtag and labels to store output.
        out = Data(tag=kwargs['tag'],
                   stack=ctx.obj['stack'],
                   comp_grid=ctx.obj['compgrid'],
                   label=kwargs['label'],
                   meta=a.meta)
        out.push(outrot, grid)
        data.add(out)
    #end

    data.deactivateAll(tag=kwargs['array'])
    data.deactivateAll(tag=kwargs['rotator'])

    vlog(ctx, 'Finishing rotation parallel to rotator array')
#end
