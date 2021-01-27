import click

from postgkyl.commands.util import vlog, pushChain

@click.command(help='Print info of active datasets.')
@click.option('-t', '--tag',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.option('-c', '--compact', is_flag=True,
              help='Show in compact mode')
@click.option('-a', '--allsets', is_flag=True,
              help='All data sets.')
@click.pass_context
def info(ctx, **kwargs):
    vlog(ctx, 'Starting info')
    pushChain(ctx, 'info', **kwargs)
    data = ctx.obj['data']
    if kwargs['allsets']:
        onlyActive = False
    else:
        onlyActive = True
    #end

    for i, dat in data.iterator(kwargs['tag'], enum=True, onlyActive=onlyActive):
        if dat.getStatus():
            color = 'green'
            bold = True
        else:
            color = None
            bold = False
        #end
        click.echo(click.style("Set {:s} ({:s}#{:d})".format(dat.getLabel(), dat.getTag(), i,),
                               fg=color, bold=bold))
        if not kwargs['compact']:
            click.echo(dat.info() + "\n")
        #end
    #end

    vlog(ctx, 'Finishing info')
#end
