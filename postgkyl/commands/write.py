import click

from postgkyl.commands.util import vlog, pushChain

@click.command(help="Write data into a file")
@click.option('-f', '--filename', type=click.STRING,
              help="Output file name")
@click.option('-t', '--txt', is_flag=True,
              help="Output file as ASCII `txt` instead of `bp`")
@click.option('-b', '--buffersize', default=1000,
              help="Set the buffer size for ADIOS write (default: 1000 MB)")
@click.pass_context
def write(ctx, **kwargs):
    vlog(ctx, 'Starting write')
    pushChain(ctx, 'write', **kwargs)

    for s in ctx.obj['sets']:
        ctx.obj['dataSets'][s].write(outName=kwargs['filename'],
                                     txt=kwargs['txt'],
                                     bufferSize=kwargs['buffersize'])
    vlog(ctx, 'Finishing write')
