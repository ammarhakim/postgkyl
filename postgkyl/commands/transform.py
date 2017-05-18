import click
import numpy as np

@click.command()
@click.argument('factor', nargs=1, type=click.FLOAT)
@click.pass_context
def mult(ctx, factor):
    for i, values in enumerate(ctx.obj['values']):
        ctx.obj['values'][i] = values * factor
