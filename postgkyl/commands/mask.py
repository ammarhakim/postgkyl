import click
import numpy as np

from postgkyl.data import GData
from postgkyl.commands.util import verb_print

@click.command(help='Mask data with specified Gkeyll mask file.')
@click.option('--use', '-u',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('--filename', '-f', type=click.STRING,
              help="Specify the file with a mask")
@click.option('--lower', '-l', type=click.FLOAT,
              help="Specify the lower theshold to be masked out.")
@click.option('--upper', '-u', type=click.FLOAT,
              help="Specify the upper theshold to be masked out.")
@click.pass_context
def mask(ctx, **kwargs):
  verb_print(ctx, 'Starting mask')
  data = ctx.obj('data')
    
  if kwargs['filename']:
    maskFld = GData(kwargs['filename']).getValues()
  #end

  for dat in data.interator(kwargs['use']):
    values = dat.getValues()

    if kwargs['filename']:
      maskFldRep = np.repeat(maskFld, dat.getNumComps(), axis=-1)
      data.setValues(np.ma.masked_where(maskFldRep < 0.0, values))
    elif kwargs['lower'] is not None and kwargs['upper'] is not None:
      dat.setValues(np.ma.masked_outside(values, kwargs['lower'], kwargs['upper']))
    elif kwargs['lower'] is not None:
      dat.setValues(np.ma.masked_less(values, kwargs['lower']))
    elif kwargs['upper'] is not None:
      dat.setValues(np.ma.masked_greater(values, kwargs['upper']))
    else:
      data.setValues(values)
      click.echo(click.style("WARNING in 'mask': No masking information specified.", fg='yellow'))
    #end
  #end
        
  verb_print(ctx, 'Finishing mask')
#end
