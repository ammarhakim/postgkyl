import click

@click.command(help='Print data info')
@click.pass_context
def info(ctx):
    numData = len(ctx.obj['values'])

    click.echo('\nPrinting info:')
    for i in range(numData):
        click.echo('Dataset #{:d}'.format(i))
        click.echo(' * Time: {:f}'.format(ctx.obj['data'][i].time))

        numDims = len(ctx.obj['values'][i].shape)
        click.echo(' * Dimensions ({:d}):'.format(numDims))
        for d in range(numDims):
            click.echo('  * Dim {:d}: Num. Cells: {:d}; Lower: {:f}; Upper: {:f}'.
              format(d, len(ctx.obj['coords'][i][d]),
                     ctx.obj['coords'][i][d][0], ctx.obj['coords'][i][d][-1]))
