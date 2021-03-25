import click

import postgkyl.diagnostics as diag
from postgkyl.commands.util import vlog, pushChain
from postgkyl.data import Data

@click.command()
@click.option('-p', '--psd', is_flag=True,
              help="Limits output to positive frequencies and returns the power spectral density |FT|^2.")
@click.option('-i', '--iso', is_flag=True,
              help="Bins power spectral density |FT|^2, making 1D power spectra from multi-dimensional data.")
@click.option('--use', '-u',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('--tag', '-t',
              help='Optional tag for the resulting array')
@click.option('--label', '-l',
              help="Custom label for the result")
@click.pass_context
def fft(ctx, **kwargs):
    """Calculate the Fourier Transform or the power-spectral density of
    input data. Only works on 1D data at present.
    """
    vlog(ctx, 'Starting FFT')
    pushChain(ctx, 'fft', **kwargs)
    data = ctx.obj['data']
    
    for dat in data.iterator(kwargs['use']):
        if kwargs['tag']:
            out = Data(tag=kwargs['tag'],
                       label=kwargs['label'],
                       compgrid=ctx.obj['compgrid'],
                       meta=dat.meta)
            grid, values = diag.fft(dat, kwargs['psd'], kwargs['iso'])
            out.push(grid, values)
            data.add(out)
        else:
            diag.fft(dat, kwargs['psd'], kwargs['iso'], overwrite=True)
        #end
    #end
        
    vlog(ctx, 'Finishing FFT')
#end
