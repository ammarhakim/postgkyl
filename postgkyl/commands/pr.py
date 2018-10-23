import click

from postgkyl.commands.util import vlog, pushChain

@click.command(help='Print the data')
@click.pass_context
def pr(ctx, **kwargs):
    vlog(ctx, 'Starting pr')
    pushChain(ctx, 'pr', **kwargs)

    for s in ctx.obj['sets']:
        print(ctx.obj['dataSets'][s].getValues())

    vlog(ctx, 'Finishing pr')
