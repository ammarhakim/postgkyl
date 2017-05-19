import click

@click.command(help='Print data info')
@click.pass_context
def info(ctx):
    numData = len(ctx.obj['values'])

    click.echo('\nPrinting info:')
    for f in range(numData):
        click.echo(' * Datastream #{:d}'.format(f))
        click.echo('  * Time: {:f}'.format(ctx.obj['data'][f].time))
        click.echo('  * Dumber of components: {:d}'.
                   format(ctx.obj['values'][f].shape[-1]))
        numDims = len(ctx.obj['values'][f].shape)-1
        click.echo('  * Dimensions ({:d}):'.format(numDims))
        for d in range(numDims):
            click.echo('   * Dim {:d}: Num. Cells: {:d}; Lower: {:f}; Upper: {:f}'.
              format(d, len(ctx.obj['coords'][f][d]),
                     ctx.obj['coords'][f][d][0], ctx.obj['coords'][f][d][-1]))
