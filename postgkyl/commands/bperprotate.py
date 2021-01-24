import click

from postgkyl.commands.util import vlog, pushChain
from postgkyl.data import Data

@click.command()
@click.option('--array', '-a',
              default='array', show_default=True,
              help="Tag for array to be rotated")
@click.option('--field', '-r',
              default='field', show_default=True,
              help="Tag for EM field data (data used for the rotation)")
@click.option('--outtag', '-o',
              default='arrayBperp', show_default=True,
              help='Tag for the resulting rotated array perpendicular to magnetic field')
@click.option('--label', '-l',
              default='arrayBperp', show_default=True,
              help="Custom label for the result")
@click.pass_context
def bperprotate(ctx, **kwargs):
    """Rotate an array perpendicular to the unit vectors of the magnetic field.
    For two arrays u and b, where b is the unit vector in the direction of the magnetic field, 
    the operation is u - (u dot b_hat) b_hat.
    """
    vlog(ctx, 'Starting rotation perpendicular to magnetic field')
    pushChain(ctx, 'arrayBpar', **kwargs)
    
    data = ctx.obj['data'] # shortcut
    
    for a, rot in zip(data.iterator(kwargs['array']),
                      data.iterator(kwargs['field'])):
        grid = a.getGrid()        
        valsarray = a.getValues()
        valsfield = field.getValues()
        # Get the components from field for the magnetic field
        B = valsfield[...,3:6]
            
        out = Data(tag=kwargs['outtag'],
                   stack=ctx.obj['stack'],
                   compgrid=ctx.obj['compgrid'],
                   label=kwargs['label'])
        out.push(postgkyl.diagnostics.perprotate(valsarray, B), grid)
        data.add(out)
    #end

    data.deactivateAll(tag=kwargs['array'])
    data.deactivateAll(tag=kwargs['field'])

    vlog(ctx, 'Finishing rotation perpendicular to magnetic field')
#end
