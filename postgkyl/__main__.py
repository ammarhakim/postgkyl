import click

from postgkyl.commands import load
from postgkyl.commands import project
from postgkyl.commands import output
from postgkyl.commands import info
from postgkyl.commands import transform
from postgkyl.data.load import GData

@click.group(chain=True)
@click.option('--filename', '-f', multiple=True)
@click.pass_context
def cli(ctx, filename):
    if filename == ():
        click.echo('No data file given. Specify file(s) with \'-f\'')
        ctx.exit()
    
    ctx.obj['files'] = filename
    ctx.obj['numFiles'] = len(filename)
    ctx.obj['data'] = []
    for i, fl in enumerate(filename):
        ctx.obj['data'].append(GData(fl))

cli.add_command(project.project)
cli.add_command(output.plot)
cli.add_command(info.info)
cli.add_command(transform.mult)

@cli.command()
@click.pass_context
def test(ctx):
    click.echo(len(ctx.obj['values']))
    #click.echo(ctx.obj['data'][0].q)

if __name__ == '__main__':
    cli(obj={})
