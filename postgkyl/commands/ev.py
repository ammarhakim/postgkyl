import click
import numpy as np
import importlib
from os import path
import sys

from postgkyl.data import GData
from postgkyl.data import select as pselect
from postgkyl.commands.util import verb_print
from postgkyl.commands import ev_cmd as cmdBase

helpStr = ""
for s in cmdBase.cmds.keys():
  helpStr += " '{:s}',".format(s)
#end

def _data(ctx, grid_stack, value_stack, ctx_stack, strIn, tags, only_active):
  strInSplit = strIn.split('[')
  if strIn[0] == 'f' or strInSplit[0] in tags:
    tag_nm = None
    if strInSplit[0] in tags:
      tag_nm = strInSplit[0]
      only_active = False
    #end
    setIdx = None
    if len(strInSplit) >= 2:
      setIdx = strInSplit[1].split(']')[0]
    #end
    compIdx = None
    if len(strInSplit) == 3:
      compIdx = strInSplit[2].split(']')[0]
    #end
    ctx_key = None
    if len(strIn.split('.')) == 2:
      ctx_key = strIn.split('.')[1]
    #end

    grid_stack.append([])
    value_stack.append([])
    ctx_stack.append([])

    for dat in ctx.obj['data'].iterator(tag=tag_nm, select=setIdx,
                                        only_active=only_active):
      tag_nm = dat.getTag()
      if ctx_key:
        grid = None
        if ctx_key in dat.ctx:
          values = np.array(dat.ctx[ctx_key])
        else:
          ctx.fail(click.style("Wrong ctx key '{:s}' specified".format(ctx_key), fg='red'))
        #end
      else:
        grid, values = pselect(dat, comp=compIdx)
      #end
      grid_stack[-1].append(grid)
      value_stack[-1].append(values)
      ctx_stack[-1].append(dat.ctx)
    #end
    return True, (tag_nm, setIdx)
  elif '(' in strIn or '[' in strIn:
    value_stack.append([eval(strIn)])
    grid_stack.append([None])
    ctx_stack.append([{}])
    return True, ()
  elif ':' in strIn or ',' in strIn:
    value_stack.append([str(strIn)])
    grid_stack.append([None])
    ctx_stack.append([{}])
    return True, ()
  else:
    try:
      value_stack.append([np.array(float(strIn))])
      grid_stack.append([None])
      ctx_stack.append([{}])
      return True, ()
    except Exception:
      return False, ()
    #end
  #end
#end

def _command(ctx, grid_stack, value_stack, ctx_stack, strIn):
  if strIn in cmdBase.cmds:
    numIn = cmdBase.cmds[strIn]['numIn']
    numOut = cmdBase.cmds[strIn]['numOut']
    func = cmdBase.cmds[strIn]['func']
  else:
    return False
  #end

  inGrid, inValues, in_ctx, numSets = [], [], [], []
  for i in range(numIn):
    inGrid.append(grid_stack.pop())
    inValues.append(value_stack.pop())
    in_ctx.append(ctx_stack.pop())
    numSets.append(len(inValues[-1]))
  #end
  for i in range(numOut):
    grid_stack.append([])
    value_stack.append([])
    ctx_stack.append([])
  #end

  for setIdx in range(max(numSets)):
    tmpGrid, tmpValues, tmp_ctx  = [], [], []
    for i in range(numIn):
      tmpGrid.append(inGrid[i][min(setIdx,numSets[i]-1)])
      tmpValues.append(inValues[i][min(setIdx,numSets[i]-1)])
      tmp_ctx.append(in_ctx[i][min(setIdx,numSets[i]-1)])
    #end
    try:
      outGrid, outValues = func(tmpGrid, tmpValues)
    except Exception as err:
      ctx.fail(click.style("{}".format(err), fg='red'))
    #end

    # Compare the ctxdata of all the inputs and copy them to a
    # ctxdata dictionary of the output
    out_ctx = {}
    for i in range(numIn):
      for k in tmp_ctx[i]:
        if k in out_ctx and tmp_ctx[i][k] == out_ctx[k]:
          pass # This key has been already copied and
               # matches the output; no action needed
        elif k in out_ctx:
          out_ctx[k] = None # There is a discrepancy between
                            # the ctxdata; set it to None
                            # and remove later
        else:
          out_ctx[k] = tmp_ctx[i][k] # Copy the ctxdata
        #end
      #end
    #end
    # Remove the discrepancies, i.e. the keys with None
    for k in out_ctx:
      if out_ctx[k] is None:
        out_ctx.pop(k)
      #end
    #end

    for i in range(numOut):
      grid_stack[-numOut+i].append(outGrid[i])
      value_stack[-numOut+i].append(outValues[i])
      ctx_stack[-numOut+i].append(out_ctx)
    #end
  #end
  return True
#end

@click.command(help="Manipulate datasets using math expressions. Expressions are specified using Reverse Polish Notation (RPN).\n Supported operators are:" + helpStr[:-1] + ". User-specifed commands can also be used.")
@click.argument('chain', nargs=1, type=click.STRING)
@click.option('--tag', '-t',
              help='Tag for the result')
@click.option('--label', '-l', show_default=True,
              help="Custom label for the result")
@click.option('--all', '-a',
              is_flag=True,
              help="Ignore the status of a dataset")
@click.pass_context
def ev(ctx, **kwargs):
  verb_print(ctx, 'Starting evaluate')
  data = ctx.obj['data']

  grid_stack, value_stack, ctx_stack = [], [], []
  chainSplit = kwargs['chain'].split(' ')
  chainSplit = list(filter(None, chainSplit))

  only_active = True
  if kwargs['all']:
    only_active = False
  #end

  tags = list(data.tagIterator(onlyActive=only_active))
  # outTag = kwargs['tag']
  # if outTag is None:
  #     if len(tags) == 1:
  #         outTag = tags[0]
  #     else:
  #         outTag = 'ev'
  #     #end
  # #end
  label = kwargs['label']
  if label is None:
    label = kwargs['chain']
  #end

  numDatasetsInChain = 0
  outDataId = ()
  for s in chainSplit:
    isData, dataId = _data(ctx, grid_stack, value_stack, ctx_stack, s, tags,
                           only_active)
    if isData and len(dataId) > 0 and dataId != outDataId:
      numDatasetsInChain += 1
      outDataId = dataId
    #end
    if not isData:
      isCommand = _command(ctx, grid_stack, value_stack, ctx_stack, s)
    #end
    if not isData and not isCommand:
      ctx.fail(click.style("Evaluate input '{:s}' represents neither data nor commad".format(s), fg='red'))
    #end
  #end

  if len(value_stack) == 0:
    ctx.fail(click.style("Evaluate stack is empty, there is nothing to return", fg='red'))
  elif len(value_stack) > 1:
    click.echo(click.style("WARNING: Length of the evaluate stack is bigger than 1, there is a posibility of unintended behavior", fg='yellow'))
  #end
  if numDatasetsInChain == 1 and kwargs['tag'] is None:
    cnt = 0
    tag = outDataId[0]
    for out in ctx.obj['data'].iterator(tag=tag, select=outDataId[1],
                                        onlyActive=only_active):
      out.push(grid_stack[-1][cnt], value_stack[-1][cnt])
      cnt += 1
    #end
  else:
    tag = outDataId[0]
    if kwargs['tag']:
      tag = kwargs['tag']
    else:
      data.deactivateAll()
    #end
    for grid, values, ctx in zip(grid_stack[-1], value_stack[-1], ctx_stack[-1]):
      out = GData(tag=tag,
                  comp_grid=ctx.obj['compgrid'],
                  label=label,
                  ctx=ctx)
      out.push(grid, values)
      data.add(out)
    #end
  #end

  verb_print(ctx, 'Finishing ev')
#end
