import click

from postgkyl.tools.stack import pushStack, pullStack, popStack

@click.command(help='Print data info')
@click.pass_context
def info(ctx):                                    
    click.echo('\nPrinting the current top of stack:')
    for s in range(ctx.obj['numSets']):
        coords, values = pullStack(ctx, s)
        click.echo(' * Dataset #{:d}'.format(s))
        click.echo('  * Time: {:f}'.format(ctx.obj['data'][s].time))
        click.echo('  * Dumber of components: {:d}'.format(values.shape[-1]))
        numDims = len(values.shape)-1
        click.echo('  * Dimensions ({:d}):'.format(numDims))
        for d in range(numDims):
            click.echo('   * Dim {:d}: Num. Cells: {:d}; Lower: {:f}; Upper: {:f}'.
                       format(d, len(coords[d]), coords[d][0], coords[d][-1]))
