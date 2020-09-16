import click

import postgkyl.diagnostics as diag
from postgkyl.commands.util import vlog, pushChain

@click.command()
@click.option('-p', '--psd', is_flag=True,
              help="Limits output to positive frequencies and returns the power spectral density |FT|^2.")
@click.pass_context
def fft(ctx, **kwargs):
    """Calculate the Fourier Transform or the power-spectral density of
    input data. Only works on 1D data at present.
    """
    vlog(ctx, 'Starting FFT')
    pushChain(ctx, 'fft', **kwargs)

    for s in ctx.obj['sets']:
        data = ctx.obj['dataSets'][s]
        diag.fft(data, kwargs['psd'], stack=True)
        
    vlog(ctx, 'Finishing FFT')
