import click

import postgkyl.diagnostics as diag
from postgkyl.commands.util import vlog, pushChain
from postgkyl.data import Data

@click.command()
@click.option('-p', '--psd', is_flag=True,
              help="Limits output to positive frequencies and returns the power spectral density |FT|^2.")
@click.option('--use', '-u',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('--tag', '-t',
              help='Optional tag for the resulting array')
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
                       compgrid=ctx.obj['compgrid'],
                       meta=dat.meta)
            grid, values = diag.fft(dat, kwargs['psd'])
            out.push(grid, values)
            data.add(out)
        else:
            diag.fft(dat, kwargs['psd'], overwrite=True)
        #end
    #end
        
    vlog(ctx, 'Finishing FFT')
#end
