import click

from postgkyl.commands.util import vlog, pushChain

@click.command(help='Print info of the current top of stack')
@click.option('-a', '--allsets', is_flag=True,
              help='All data sets')
@click.pass_context
def info(ctx, **kwargs):
    vlog(ctx, 'Starting info')
    pushChain(ctx, 'info.info', **kwargs)

    if kwargs['allsets'] is True:
        vlog(ctx, ("Printing the current top of stack "
                   "information (all data sets):"))
        sets = range(ctx.obj['numSets'])
    else:
        vlog(ctx, ("Printing the current top of stack "
                   "information (active data sets):"))
        sets = ctx.obj['sets']
        
    for s in sets:
        click.echo("Dataset #{:d}".format(ctx.obj['setIds'][s]))
        click.echo(ctx.obj['dataSets'][s].info() + "\n")

    vlog(ctx, 'Finishing info')
