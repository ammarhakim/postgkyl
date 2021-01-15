import click

from postgkyl.commands.util import vlog, pushChain

@click.command(help='Print info of active datasets.')
@click.option('-t', '--tag',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('-a', '--allsets', is_flag=True,
              help='All data sets.')
@click.pass_context
def info(ctx, **kwargs):
    vlog(ctx, 'Starting info')
    pushChain(ctx, 'info', **kwargs) 

    for dat in ctx.obj['data'].iterator(kwargs['tag']):
        click.echo("Dataset")
        click.echo(dat.info() + "\n")
    #end

    vlog(ctx, 'Finishing info')
#end
