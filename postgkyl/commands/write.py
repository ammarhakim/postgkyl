import click

from postgkyl.commands.util import vlog, pushChain

@click.command(help="Write data into a file")
@click.option('-f', '--filename', type=click.STRING,
              help="Output file name.")
@click.option('-t', '--txt', is_flag=True,
              help="Output file as ASCII `txt` instead of `bp`.")
@click.pass_context
def write(ctx, **kwargs):
    vlog(ctx, 'Starting write')
    pushChain(ctx, 'write', **kwargs)

    for s in ctx.obj['sets']:
        ctx.obj['dataSets'][s].write(fName=kwargs['filename'],
                                     txt=kwargs['txt'])
    vlog(ctx, 'Finishing write')
