import click
import numpy as np
import importlib
from os import path
import sys

from postgkyl.data import Data
from postgkyl.data import select as pselect
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

def _data(ctx, gridStack, valueStack, metaStack, strIn, tags):
    if strIn[0] == 'f' or strIn.split('[')[0] in tags:
        splits = strIn.split('[')
        if splits[0] in tags:
            tagNm = splits[0]
        elif strIn[0] == 'f' and len(tags) == 1:
            tagNm = tags[0]
        else:
            ctx.fail(click.style("'f' cannot be used if there is more then one active tags ({:d} active); use tag names instead.".format(len(tags)), fg='red'))
        #end
        setIdx = None
        if len(splits) >= 2:
            setIdx = splits[1].split(']')[0]
        #end
        compIdx = None
        if len(splits) == 3:
            compIdx = splits[2].split(']')[0]
        #end

        metaKey = None
        if len(strIn.split('.')) == 2:
            metaKey = strIn.split('.')[1]
        #end

        gridStack.append([])
        valueStack.append([])
        metaStack.append([])
        for dat in ctx.obj['data'].iterator(tag=tagNm, select=setIdx, onlyActive=False):
            if metaKey:
                grid = None
                if metaKey in dat.meta:
                    values = np.array(dat.meta[metaKey])
                else:
                    ctx.fail(click.echo("Wrong meta key '{:s}' specified".format(metaKey), fg='red'))
                #end
            else:
                grid, values = pselect(dat, comp=compIdx)
            #end
            gridStack[-1].append(grid)
            valueStack[-1].append(values)
            metaStack[-1].append(dat.meta)
        #end
        return True
    elif '(' in strIn or '[' in strIn:
        valueStack.append([eval(strIn)])
        gridStack.append([None])
        metaStack.append([{}])
        return True
    elif ':' in strIn or ',' in strIn:
        valueStack.append([str(strIn)])
        gridStack.append([None])
        metaStack.append([{}])
        return True
    else:
        try:
            valueStack.append([np.array(float(strIn))])
            gridStack.append([None])
            metaStack.append([{}])
            return True
        except Exception:
            return False
        #end
    #end
#end

def _command(ctx, gridStack, valueStack, metaStack, strIn):
    if userCommands and strIn in cmdUser.cmds:
        numIn = cmdUser.cmds[strIn]['numIn']
        numOut = cmdUser.cmds[strIn]['numOut']
        func = cmdUser.cmds[strIn]['func']
    elif strIn in cmdBase.cmds:
        numIn = cmdBase.cmds[strIn]['numIn']
        numOut = cmdBase.cmds[strIn]['numOut']
        func = cmdBase.cmds[strIn]['func']
    else:
        return False
    #end
    
    inGrid, inValues, inMeta, numSets = [], [], [], []
    for i in range(numIn):
        inGrid.append(gridStack.pop())
        inValues.append(valueStack.pop())
        inMeta.append(metaStack.pop())
        numSets.append(len(inValues[-1]))
    #end
    for i in range(numOut):
        gridStack.append([])
        valueStack.append([])
        metaStack.append([])
    #end

    for setIdx in range(max(numSets)):
        tmpGrid, tmpValues, tmpMeta  = [], [], []
        for i in range(numIn):
            tmpGrid.append(inGrid[i][min(setIdx,numSets[i]-1)])
            tmpValues.append(inValues[i][min(setIdx,numSets[i]-1)])
            tmpMeta.append(inMeta[i][min(setIdx,numSets[i]-1)])
        #end
        try:
            outGrid, outValues = func(tmpGrid, tmpValues)
        except Exception as err:
            ctx.fail(click.style("{}".format(err), fg='red'))
        #end

        # Compare the metadata of all the inputs and copy them to a
        # metadata dictionary of the output
        outMeta = {}
        for i in range(numIn):
            for k in tmpMeta[i]:
                if k in outMeta and tmpMeta[i][k] == outMeta[k]:
                    pass # This key has been already copied and
                         # matches the output; no action needed
                elif k in outMeta:
                    outMeta[k] = None # There is a discrepancy between
                                      # the metadata; set it to None
                                      # and remove later
                else:
                    outMeta[k] = tmpMeta[i][k] # Copy the metadata
                #end
            #end
        #end
        # Remove the discrepancies, i.e. the keys with None
        keys = list(outMeta)
        for k in keys:
            if outMeta[k] is None:
                outMeta.pop(k)
            #end
        #end
        
        for i in range(numOut):
            gridStack[-numOut+i].append(outGrid[i])
            valueStack[-numOut+i].append(outValues[i])
            metaStack[-numOut+i].append(outMeta)
        #end
    #end
    return True
#end

@click.command(help="Manipulate datasets using math expressions. Expressions are specified using Reverse Polish Notation (RPN).\n Supported operators are:" + helpStr[:-1] + ". User-specifed commands can also be used.")
@click.argument('chain', nargs=1, type=click.STRING)
@click.option('--tag', '-t',
              help='Tag for the result')
@click.option('--label', '-l',
              default='ev', show_default=True,
              help="Custom label for the result")
@click.pass_context
def ev(ctx, **kwargs):
    vlog(ctx, 'Starting evaluate')
    pushChain(ctx, 'ev', **kwargs)
    data = ctx.obj['data']
    
    gridStack, valueStack, metaStack = [], [], []
    chainSplit = kwargs['chain'].split(' ')
    chainSplit = list(filter(None, chainSplit))

    tags = list(data.tagIterator(onlyActive=False))
    outTag = kwargs['tag']
    if outTag is None:
        if len(tags) == 1:
            outTag = tags[0]
        else:
            outTag = 'ev'
        #end
    #end
            
    for s in chainSplit:
        isData = _data(ctx, gridStack, valueStack, metaStack, s, tags)
        if not isData:
            isCommand = _command(ctx, gridStack, valueStack, metaStack, s)
        #end
        if not isData and not isCommand:
            ctx.fail(click.style("Evaluate input '{:s}' represents neither data nor commad".format(s), fg='red'))
        #end
    #end

    if len(valueStack) == 0:
        ctx.fail(click.style("Evaluate stack is empty, there is nothing to return".format(s), fg='red'))
    elif len(valueStack) > 1:
        click.echo(click.style("WARNING: Length of the evaluate stack is bigger than 1, there is a posibility of unintended behavior".format(s), fg='yellow'))
    #end

    data.deactivateAll()

    for grid, values, meta in zip(gridStack[-1], valueStack[-1], metaStack[-1]):
        out = Data(tag=outTag,
                   compgrid=ctx.obj['compgrid'],
                   label=kwargs['label'],
                   meta=meta)
        out.push(grid, values)
        data.add(out)
    #end

    vlog(ctx, 'Finishing ev')
#end
