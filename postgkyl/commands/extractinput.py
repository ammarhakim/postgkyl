import click
import base64

from postgkyl.commands.util import vlog, pushChain

@click.command(help='Extract embedded input file from compatible BP files')
@click.pass_context
def extractinput(ctx, **kwargs):
    vlog(ctx, 'Starting ')
    pushChain(ctx, 'extractinput', **kwargs)

    sets = ctx.obj['sets']
        
    for s in sets:
        pass

    vlog(ctx, 'Finishing extractinput')
