import click

from postgkyl.commands.util import vlog, pushChain

@click.command()
@click.option('-f', '--filename', type=click.STRING,
              help="Output file name")
@click.option('-t', '--txt', is_flag=True,
              help="Output file as ASCII `txt` instead of `bp`")
@click.option('-b', '--buffersize', default=1000,
              help="Set the buffer size for ADIOS write (default: 1000 MB)")
@click.pass_context
def write(ctx, **kwargs):
    """Write active dataset to a file. The output file is either a text
    file (use only for small data) or a ADIOS BP file. If data is
    saved as BP file it can be later loaded back into pgkyl to further
    manipulate or plot it.

    """
    vlog(ctx, 'Starting write')
    pushChain(ctx, 'write', **kwargs)

    for s in ctx.obj['sets']:
        ctx.obj['dataSets'][s].write(outName=kwargs['filename'],
                                     txt=kwargs['txt'],
                                     bufferSize=kwargs['buffersize'])
    vlog(ctx, 'Finishing write')
