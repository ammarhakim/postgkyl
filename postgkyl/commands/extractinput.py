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
        encInp = ctx.obj['dataSets'][s].inputfile
        if encInp:
            inpfile = base64.decodebytes(encInp.encode('utf-8')).decode('utf-8')
            click.echo(inpfile)
        else:
            click.echo("No embedded input file!")

    vlog(ctx, 'Finishing extractinput')
