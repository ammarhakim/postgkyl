import click
import numpy as np
import importlib
from os import path
import sys

from postgkyl.data import GData
from postgkyl.commands.util import vlog, pushChain

# load base commands
from  postgkyl.commands import ev_cmd as cmdBase
# load user commands
if path.isfile(path.expanduser("~") + '/pgkyl_ev.py'):
    sys.path.insert(0, path.expanduser("~"))
    import pgkyl_ev as cmdUser
    userCommands = True
else:
    userCommands = False

helpStr = ""
for s in cmdBase.cmds.keys():
    helpStr += " '{:s}',".format(s)
if userCommands:
    for s in cmdUser.cmds.keys():
        helpStr += " '{:s}',".format(s)


def _data(ctx, gridStack, evalStack, s):
    if s[0] == 'f':
        try:
            if '[' in s:
                setIdx = s[1:].split('[')[0]
                compIdx = int(s.split('[')[1].split(']')[0])
            else:
                if len(s) == 1:
                    setIdx = 0
                else:
                    setIdx = s[1:]
                compIdx = None
        except ValueError:
            click.echo(click.style("ERROR in 'ev': Data set name '{:s}' is not in right format. It needs to be 'f#', f#[#]', 'f*', or 'f*[#]'".format(s), fg='red'))
            ctx.exit()
            
        if setIdx == '*':
            if len(gridStack) == 1: # switching stacks to "piece-wise" mode
                for i in range(len(ctx.obj['sets'])-1):
                    gridStack.append(list(gridStack[0]))
                    evalStack.append(list(evalStack[0]))

            for i, setIdx in enumerate(ctx.obj['sets']):
                gridStack[i].append(ctx.obj['dataSets'][setIdx].getGrid())
                values = ctx.obj['dataSets'][setIdx].getValues()
                if compIdx is not None:
                    values = values[..., compIdx, np.newaxis]
                evalStack[i].append(values)
        else:
            try:
                setIdx = int(setIdx)
            except ValueError:
                click.echo(click.style("ERROR in 'ev': Data set name '{:s}' is not in right format. It needs to be 'f#', f#[#]', 'f*', or 'f*[#]'".format(s), fg='red'))
                ctx.exit()

            if setIdx < 0 or setIdx >= len(ctx.obj['sets']):
                click.echo(click.style("ERROR in 'ev': Data set index '{:d}' needs to be between 0 and {:d}".format(setIdx,  len(ctx.obj['sets'])-1), fg='red'))
                ctx.exit()
            
            for i in range(len(gridStack)):
                gridStack[i].append(ctx.obj['dataSets'][ctx.obj['sets'][setIdx]].getGrid())
                values = ctx.obj['dataSets'][ctx.obj['sets'][setIdx]].getValues()
                if compIdx is not None:
                    values = values[..., compIdx, np.newaxis]
                evalStack[i].append(values)
        return True
    else:
        try:
            num = float(s)
            for i in range(len(gridStack)):
                evalStack[i].append(num)
                gridStack[i].append([])
            return True
        except Exception:
            return False


def _command(gridStack, evalStack, s):
    if userCommands and s in cmdUser.cmds:
        numIn = cmdUser.cmds[s]['numIn']
        numOut = cmdUser.cmds[s]['numOut']
        func = cmdUser.cmds[s]['func']
    elif s in cmdBase.cmds:
        numIn = cmdBase.cmds[s]['numIn']
        numOut = cmdBase.cmds[s]['numOut']
        func = cmdBase.cmds[s]['func']
    else:
        return False

    for i in range(len(evalStack)):
        inGrid, inValues = [], []
        for j in range(numIn):
            inGrid.append(gridStack[i].pop())
            inValues.append(evalStack[i].pop())
        outGrid, outValues = func(inGrid, inValues)
        for j in range(numOut):
            gridStack[i].append(outGrid[j])
            evalStack[i].append(outValues[j])

    return True

@click.command(help="Evaluate stuff using Reverse Polish Notation (RPN).\n Supported operators are:" + helpStr[:-1])
@click.argument('chain', nargs=1, type=click.STRING)
@click.pass_context
def ev(ctx, **kwargs):
    vlog(ctx, 'Starting evaluate')
    pushChain(ctx, 'ev', **kwargs)

    gridStack, evalStack = [[]], [[]]
    chainSplit = kwargs['chain'].split(' ')
    chainSplit = list(filter(None, chainSplit))

    for s in chainSplit:
        isData = _data(ctx, gridStack, evalStack, s)
        if not isData:
            isCommand = _command(gridStack, evalStack, s)

        if not isData and not isCommand:
            click.echo(click.style("ERROR in 'ev': Evaluate input '{:s}' represents neither data nor commad".format(s), fg='red'))
            ctx.exit()

    if len(evalStack[0]) == 0:
        click.echo(click.style("ERROR in 'ev': Evaluate stack is empty, there is nothing to return".format(s), fg='red'))
        ctx.exit()
    elif len(evalStack[0]) > 1:
        click.echo(click.style("WARNING in 'ev': Length of the evaluate stack is bigger than 1, there is a posibility of unintended behavior".format(s), fg='yellow'))
    if len(evalStack) == 1:
        vlog(ctx, 'ev: Creating new dataset')
        idx = len(ctx.obj['dataSets'])
        ctx.obj['setIds'].append(idx)
        ctx.obj['dataSets'].append(GData())
        ctx.obj['dataSets'][idx].pushGrid(gridStack[0][-1])
        ctx.obj['dataSets'][idx].pushValues(evalStack[0][-1])
        ctx.obj['dataSets'][idx].name = 'ev'

        vlog(ctx, 'ev: Active data set switched to #{:d}'.format(idx))
        ctx.obj['sets'] = [idx]
    else:
        for i, setIdx in enumerate(ctx.obj['sets']):
            ctx.obj['dataSets'][setIdx].pushGrid(gridStack[i][-1])
            ctx.obj['dataSets'][setIdx].pushValues(evalStack[i][-1])

    vlog(ctx, 'Finishing ev')
