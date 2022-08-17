import click
import shutil

from postgkyl.commands.util import vlog, pushChain

@click.command()
@click.option('--use', '-u',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('-f', '--filename', type=click.STRING,
              help="Output file name(s)")
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
    file_names = kwargs['filename'].split(',')
    append = False
    cleaning = True
    
    for i, dat in data.iterator(kwargs['use'],
                                enum=True):
      if i < len(file_names):
        fn = file_names[i]
      else:
        fn = '{:s}{:d}.bp'.format(file_names[0].split('.')[0], i)
      #end
      
      if kwargs['single']:
        var_name = '{:s}{:d}'.format(dat.getTag(), i)
        fn = file_names[0]
        cleaning = False
      #end
      
      dat.write(out_name=fn,
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
      shutil.move(fn + '.dir/' + nm + '.0', fn)
      shutil.rmtree(fn + '.dir')
    #end
    vlog(ctx, 'Finishing write')
#end
