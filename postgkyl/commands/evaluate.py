import click
import numpy as np

from postgkyl.data import GData
from postgkyl.commands.util import vlog, pushChain

def _data(ctx, gridStack, evalStack, s):
    if s[0] == 'f':
        if '[' in s:
            setIdx = int(s[1:].split('[')[0])
            compIdx = int(s.split('[')[1].split(']')[0])
        else:
            setIdx = int(s[1:])
            compIdx = None
        gridStack.append(ctx.obj['dataSets'][setIdx].getGrid())
        values = ctx.obj['dataSets'][setIdx].getValues()
        if compIdx is not None:
            values = values[..., compIdx, np.newaxis]
        evalStack.append(values)
        return True
    else:
        try:
            num = float(s)
            evalStack.append(num)
            gridStack.append(None)
        except Exception:
            return False
        return True

def _command(gridStack, evalStack, s):
    if s == '+':
        v1 = evalStack.pop()
        v0 = evalStack.pop()
        evalStack.append(v0+v1)
        g1 = gridStack.pop()
        g0 = gridStack.pop()
        if g0 is not None:
            gridStack.append(g0)
        else:
            gridStack.append(g1)
        return True
    if s == '-':
        v1 = evalStack.pop()
        v0 = evalStack.pop()
        evalStack.append(v0-v1)
        g1 = gridStack.pop()
        g0 = gridStack.pop()
        if g0 is not None:
            gridStack.append(g0)
        else:
            gridStack.append(g1)
        return True
    if s == '*':
        v1 = evalStack.pop()
        v0 = evalStack.pop()
        evalStack.append(v0*v1)
        g1 = gridStack.pop()
        g0 = gridStack.pop()
        if g0 is not None:
            gridStack.append(g0)
        else:
            gridStack.append(g1)
        return True
    if s == '/':
        v1 = evalStack.pop()
        v0 = evalStack.pop()
        evalStack.append(v0/v1)
        g1 = gridStack.pop()
        g0 = gridStack.pop()
        if g0 is not None:
            gridStack.append(g0)
        else:
            gridStack.append(g1)
        return True
    else:
        return False

@click.command(help='Evaluate stuff using Reverse Polish Notation (RPN)')
@click.argument('chain', nargs=1, type=click.STRING)
@click.pass_context
def evaluate(ctx, **kwargs):
    vlog(ctx, 'Starting evaluate')
    pushChain(ctx, 'evaluate', **kwargs)

    gridStack, evalStack = [], []
    chainSplit = kwargs['chain'].split(' ')

    for s in chainSplit:
        isData = _data(ctx, gridStack, evalStack, s)
        if not isData:
            isCommand = _command(gridStack, evalStack, s)

        if not isData and not isCommand:
            click.echo(click.style("ERROR: Evaluate input '{:s}' represents neither data nor commad".format(s), fg='red'))
            ctx.exit()

    if len(evalStack) == 0:
        click.echo(click.style("ERROR: Evaluate stack is empty, there is nothing to return".format(s), fg='red'))
        ctx.exit()
    elif len(evalStack) > 1:
        click.echo(click.style("WARNING: Evaluate stack length is larger than one, there is a posibility of unintended behavior".format(s), fg='yellow'))

    vlog(ctx, 'evaluate: Creating new dataset')
    idx = len(ctx.obj['dataSets'])
    ctx.obj['setIds'].append(idx)
    ctx.obj['dataSets'].append(GData())
    ctx.obj['dataSets'][idx].pushGrid(gridStack[-1])
    ctx.obj['dataSets'][idx].pushValues(evalStack[-1])
    ctx.obj['dataSets'][idx].name = 'ev'

    vlog(ctx, 'evaluate: Active data set switched to #{:d}'.format(idx))
    ctx.obj['sets'] = [idx]

    vlog(ctx, 'Finishing evaluate')
