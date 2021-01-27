import click

@click.command(help='Pop the data stack')
@click.option('--tag', '-t',
              help='Specify a \'tag\' to apply to (default all tags).')
@click.pass_context
def pop(ctx):
    vlog(ctx, 'Poping the stack')
    pushChain(ctx, 'pop')
    data = ctx.obj['data']
    for dat in data.iterator(kwargs['tag']):
        dat.pop()
    #end
#end
