import click
import shutil

from postgkyl.commands.util import vlog, pushChain

@click.command()
@click.option('--use', '-u',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('-f', '--filename', type=click.STRING, prompt=True,
              help="Output file name")
@click.option('-m', '--mode', type=click.Choice(['bp', 'txt', 'npy']),
              default='bp', 
              help="Output file mode. One of `bp` (ADIOS BP file; default), `txt` (ASCII text file), or `npy` (NumPy binary file)")
@click.option('-s', '--single', is_flag=True,
              help='Write all dataset into one file')
@click.option('-b', '--buffersize', default=1000,
              help='Set the buffer size for ADIOS write (default: 1000 MB)')
@click.pass_context
def write(ctx, **kwargs):
    """Write active dataset to a file. The output file format can be set 
    with ``--mode``, and is ADIOS BP by default. If data is
    saved as BP file it can be later loaded back into pgkyl to further
    manipulate or plot it.

    """
    vlog(ctx, 'Starting write')
    pushChain(ctx, 'write', **kwargs)
    data = ctx.obj['data']
    
    var_name = None
    append = False
    cleaning = True
    fn = kwargs['filename']
    mode = kwargs['mode']
    if len(fn.split('.')) > 1:
      mode = str(fn.split('.')[-1])
      fn =  str(fn.split('.')[0])
    #end
    
    num_files = data.getNumDatasets(tag=kwargs['use'])
    for i, dat in data.iterator(tag=kwargs['use'],
                                enum=True):
      out_name = '{:s}.{:s}'.format(fn, mode)
      if kwargs['single']:
        var_name = '{:s}_{:d}'.format(dat.getTag(), i)
        cleaning = False
      else:
        if num_files > 1:
          out_name = '{:s}_{:d}.{:s}'.format(fn, i, mode)
        #end
      #end
      
      dat.write(out_name=out_name,
                mode=kwargs['mode'],
                bufferSize=kwargs['buffersize'],
                append=append,
                var_name=var_name,
                cleaning=cleaning)

      if kwargs['single']:
        append = True
      #end
    #end

    # Cleaning
    if not cleaning:
      if len(fn.split('/')) > 1:
        nm = fn.split('/')[-1]
      else:
        nm = fn
      #end
      shutil.move('{:s}.{:s}.dir/{:s}.{:s}.0'.format(fn, mode, fn, mode),
                  '{:s}.{:s}'.format(fn, mode))
      shutil.rmtree('{:s}.{:s}.dir'.format(fn, mode))
    #end
    vlog(ctx, 'Finishing write')
#end
