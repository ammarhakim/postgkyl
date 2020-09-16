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
#end

helpStr = ""
for s in cmdBase.cmds.keys():
    helpStr += " '{:s}',".format(s)
#end
if userCommands:
    for s in cmdUser.cmds.keys():
        helpStr += " '{:s}',".format(s)
    #end
#end

def _copyMeta(meta, data):
    if meta["isModal"] is None:
        meta["isModal"] = data.isModal
    elif meta["isModal"] == data.isModal:
        pass
    else:
        meta["isModal"] = False
    #end
    if meta["polyOrder"] is None:
        meta["polyOrder"] = data.polyOrder
    elif meta["polyOrder"] == data.polyOrder:
        pass
    else:
        meta["polyOrder"] = 'mixed'
    #end
    if meta["basisType"] is None:
        meta["basisType"] = data.basisType
    elif meta["basisType"] == data.basisType:
        pass
    else:
        meta["basisType"] = 'unknown'
    #end
    return meta
#end

def _data(ctx, gridStack, evalStack, s, meta):
    if s[0] == 'f':
        try:
            if '[' in s:
                setIdx = s[1:].split('[')[0]
                compIdx = s.split('[')[1].split(']')[0]
                if ':' in compIdx:
                    tmp = compIdx.split(':')
                    if tmp[0] == '':
                        lo = None
                    else:
                        lo = int(tmp[0])
                    if tmp[1] == '':
                        up = None
                    else:
                        up = int(tmp[1])
                    compIdx = slice(lo, up)
                else:
                    compIdx = int(compIdx)
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
                #end
            #end
            for i, setIdx in enumerate(ctx.obj['sets']):
                gridStack[i].append(ctx.obj['dataSets'][setIdx].getGrid())
                values = ctx.obj['dataSets'][setIdx].getValues()
                _copyMeta(meta, ctx.obj['dataSets'][setIdx])

                if compIdx is not None:
                    if type(compIdx) == int:
                        values = values[..., compIdx, np.newaxis]
                    else:
                        values = values[..., compIdx]
                    #end
                evalStack[i].append(values)
            #end
        else:
            try:
                setIdx = int(setIdx)
            except ValueError:
                click.echo(click.style("ERROR in 'ev': Data set name '{:s}' is not in right format. It needs to be 'f#', f#[#]', 'f*', or 'f*[#]'".format(s), fg='red'))
                ctx.exit()

            if setIdx >= len(ctx.obj['sets']):
                click.echo(click.style("ERROR in 'ev': Data set index '{:d}' needs to be between 0 and {:d}".format(setIdx,  len(ctx.obj['sets'])-1), fg='red'))
                ctx.exit()
            #end

            if setIdx < 0:
                setIdx = len(ctx.obj['sets']) + setIdx
            #end
            
            for i in range(len(gridStack)):
                gridStack[i].append(ctx.obj['dataSets'][ctx.obj['sets'][setIdx]].getGrid())
                values = ctx.obj['dataSets'][ctx.obj['sets'][setIdx]].getValues()
                if compIdx is not None:
                    if type(compIdx) == int:
                        values = values[..., compIdx, np.newaxis]
                    else:
                        values = values[..., compIdx]
                    #end
                #end
                evalStack[i].append(values)
                _copyMeta(meta, ctx.obj['dataSets'][ctx.obj['sets'][setIdx]])
            #end

        return True
    elif '(' in s or '[' in s:
        for i in range(len(gridStack)):
            evalStack[i].append(eval(s))
            gridStack[i].append([])
        #end
        return True
    elif ':' in s or ',' in s:
        for i in range(len(gridStack)):
            evalStack[i].append(str(s))
            gridStack[i].append([])
        #end
        return True
    else:
        try:
            num = float(s)
            for i in range(len(gridStack)):
                evalStack[i].append(num)
                gridStack[i].append([])
            #end
            return True
        except Exception:
            return False
        #end
    #end
#end

def _command(ctx, gridStack, evalStack, s):
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
    #end

    for i in range(len(evalStack)):
        inGrid, inValues = [], []
        for j in range(numIn):
            inGrid.append(gridStack[i].pop())
            inValues.append(evalStack[i].pop())
        #end
        try:
            outGrid, outValues = func(inGrid, inValues)
        except Exception as err:
            click.echo(click.style("{}".format(err), fg='red'))
            ctx.exit()
        #end
        for j in range(numOut):
            gridStack[i].append(outGrid[j])
            evalStack[i].append(outValues[j])
        #end
    #end
    return True
#end

@click.command(help="Manipulate dataset using math expressions. Expressions are specified using Reverse Polish Notation (RPN).\n Supported operators are:" + helpStr[:-1] + ". User-specifed commands can also be used.")
@click.argument('chain', nargs=1, type=click.STRING)
@click.option('--label', '-l',
              help="Specify a custom label for the dataset resulting from the expression.")
@click.pass_context
def ev(ctx, **kwargs):
    vlog(ctx, 'Starting evaluate')
    pushChain(ctx, 'ev', **kwargs)

    gridStack, evalStack = [[]], [[]]
    chainSplit = kwargs['chain'].split(' ')
    chainSplit = list(filter(None, chainSplit))
    meta = {
        "isModal" : None,
        "polyOrder" : None,
        "basisType" : None
    }

    for s in chainSplit:
        isData = _data(ctx, gridStack, evalStack, s, meta)
        if not isData:
            isCommand = _command(ctx, gridStack, evalStack, s)
        #end
        if not isData and not isCommand:
            click.echo(click.style("ERROR in 'ev': Evaluate input '{:s}' represents neither data nor commad".format(s), fg='red'))
            ctx.exit()
        #end
    #end

    if len(evalStack[0]) == 0:
        click.echo(click.style("ERROR in 'ev': Evaluate stack is empty, there is nothing to return".format(s), fg='red'))
        ctx.exit()
    elif len(evalStack[0]) > 1:
        click.echo(click.style("WARNING in 'ev': Length of the evaluate stack is bigger than 1, there is a posibility of unintended behavior".format(s), fg='yellow'))
    #end
    if len(evalStack) == 1:
        vlog(ctx, 'ev: Creating new dataset')
        idx = len(ctx.obj['dataSets'])
        ctx.obj['setIds'].append(idx)
        ctx.obj['dataSets'].append(GData())
        ctx.obj['dataSets'][idx].pushGrid(gridStack[0][-1])
        ctx.obj['dataSets'][idx].pushValues(evalStack[0][-1])
        ctx.obj['dataSets'][idx].name = 'ev'
        ctx.obj['dataSets'][idx].isModal = meta['isModal']
        #end
        if meta['polyOrder'] is not 'mixed':
            ctx.obj['dataSets'][idx].polyOrder = meta['polyOrder']
        #end
        if meta['basisType'] is not 'mixed':
            ctx.obj['dataSets'][idx].basisType = meta['basisType']
        #end

        vlog(ctx, 'ev: Active data set switched to #{:d}'.format(idx))
        ctx.obj['sets'] = [idx]

        if kwargs['label']:
            ctx.obj['labels'].append(kwargs['label'])
        else:
            ctx.obj['labels'].append(kwargs['chain'])
        #end
    else:
        for i, setIdx in enumerate(ctx.obj['sets']):
            ctx.obj['dataSets'][setIdx].pushGrid(gridStack[i][-1])
            ctx.obj['dataSets'][setIdx].pushValues(evalStack[i][-1])
        #end
    #end

    vlog(ctx, 'Finishing ev')
#end
