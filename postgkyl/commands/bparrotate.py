import click

from postgkyl.commands.util import vlog, pushChain
from postgkyl.data import GData
import postgkyl.diagnostics as diag

@click.command()
@click.option('--array', '-a',
              default='array', show_default=True,
              help="Tag for array to be rotated")
@click.option('--field', '-r',
              default='field', show_default=True,
              help="Tag for EM field data (data used for the rotation)")
@click.option('--tag', '-t',
              default='arrayBpar', show_default=True,
              help='Tag for the resulting rotated array parallel to magnetic field')
@click.option('--label', '-l',
              default='arrayBpar', show_default=True,
              help="Custom label for the result")
@click.pass_context
def bparrotate(ctx, **kwargs):
    """Rotate an array parallel to the unit vectors of the magnetic field.
    For two arrays u and b, where b is the unit vector in the direction of the magnetic field, 
    the operation is (u dot b_hat) b_hat. Note that the magnetic field is a three-component field, 
    so the output is a new vector whose components are (u_{b_x}, u_{b_y}, u_{b_z}), i.e.,
    the x, y, and z components of the vector u parallel to the magnetic field. 
    """
    vlog(ctx, 'Starting rotation parallel to magnetic field')
    pushChain(ctx, 'arrayBpar', **kwargs)
    
    data = ctx.obj['data'] # shortcut
    
    for a, rot in zip(data.iterator(kwargs['array']),
                      data.iterator(kwargs['field'])):
        # Magnetic field is components 3, 4, & 5 in field array
        grid, outrot = diag.parrotate(a, rot, '3:6')
        # Create new GData structure with appropriate outtag and labels to store output.
        out = GData(tag=kwargs['tag'],
                    comp_grid=ctx.obj['compgrid'],
                    label=kwargs['label'],
                    meta=a.meta)
        out.push(grid, outrot)
        data.add(out)
    #end

    data.deactivateAll(tag=kwargs['array'])
    data.deactivateAll(tag=kwargs['field'])

    vlog(ctx, 'Finishing rotation parallel to magnetic field')
#end
