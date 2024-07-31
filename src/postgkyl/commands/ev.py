import click
import numpy as np

from postgkyl.commands import ev_cmd as cmd_base
from postgkyl.data import GData
from postgkyl.data import select as pselect
from postgkyl.utils import verb_print


help_str = ""
for s in cmd_base.cmds.keys():
  help_str += f" '{s:s}',"
# end


def _data(ctx, grid_stack, value_stack, ctx_stack, str_in, tags, only_active):
  str_in_split = str_in.split("[")
  if str_in[0] == "f" or str_in_split[0] in tags:
    tag_nm = None
    if str_in_split[0] in tags:
      tag_nm = str_in_split[0]
      only_active = False
    # end
    set_idx = None
    if len(str_in_split) >= 2:
      set_idx = str_in_split[1].split("]")[0]
    # end
    comp_idx = None
    if len(str_in_split) == 3:
      comp_idx = str_in_split[2].split("]")[0]
    # end
    ctx_key = None
    if len(str_in.split(".")) == 2:
      ctx_key = str_in.split(".")[1]
    # end

    grid_stack.append([])
    value_stack.append([])
    ctx_stack.append([])

    for dat in ctx.obj["data"].iterator(tag=tag_nm, select=set_idx, only_active=only_active):
      tag_nm = dat.get_tag()
      if ctx_key:
        grid = None
        if ctx_key in dat.ctx:
          values = np.array(dat.ctx[ctx_key])
        else:
          ctx.fail(click.style(f"Wrong ctx key '{ctx_key:s}' specified", fg="red"))
        # end
      else:
        grid, values = pselect(dat, comp=comp_idx)
      # end
      grid_stack[-1].append(grid)
      value_stack[-1].append(values)
      ctx_stack[-1].append(dat.ctx)
    # end
    return True, (tag_nm, set_idx)
  elif "(" in str_in or "[" in str_in:
    value_stack.append([eval(str_in)])
    grid_stack.append([None])
    ctx_stack.append([{}])
    return True, ()
  elif ":" in str_in or "," in str_in:
    value_stack.append([str(str_in)])
    grid_stack.append([None])
    ctx_stack.append([{}])
    return True, ()
  else:
    try:
      value_stack.append([np.array(float(str_in))])
      grid_stack.append([None])
      ctx_stack.append([{}])
      return True, ()
    except Exception:
      return False, ()
    # end
  # end


def _compare(a, b) -> bool:
  if isinstance(a, np.ndarray):
    return np.array_equal(a, b)
  else:
    return a == b
  # end


def _command(ctx, grid_stack, value_stack, ctx_stack, str_in):
  if str_in in cmd_base.cmds:
    num_in = cmd_base.cmds[str_in]["num_in"]
    num_out = cmd_base.cmds[str_in]["num_out"]
    func = cmd_base.cmds[str_in]["func"]
  else:
    return False
  # end

  in_grid, in_values, in_ctx, num_sets = [], [], [], []
  for i in range(num_in):
    in_grid.append(grid_stack.pop())
    in_values.append(value_stack.pop())
    in_ctx.append(ctx_stack.pop())
    num_sets.append(len(in_values[-1]))
  # end
  for i in range(num_out):
    grid_stack.append([])
    value_stack.append([])
    ctx_stack.append([])
  # end

  for set_idx in range(max(num_sets)):
    tmp_grid, tmp_values, tmp_ctx = [], [], []
    for i in range(num_in):
      tmp_grid.append(in_grid[i][min(set_idx, num_sets[i] - 1)])
      tmp_values.append(in_values[i][min(set_idx, num_sets[i] - 1)])
      tmp_ctx.append(in_ctx[i][min(set_idx, num_sets[i] - 1)])
    # end
    try:
      out_grid, out_values = func(tmp_grid, tmp_values)
    except Exception as err:
      ctx.fail(click.style(f"{err}", fg="red"))
    # end

    # Compare the ctx data of all the inputs and copy them to a
    # ctx data dictionary of the output
    out_ctx = {}
    remove_list = []
    for i in range(num_in):
      for key in tmp_ctx[i]:
        if key in out_ctx and _compare(tmp_ctx[i][key], out_ctx[key]):  # tmp_ctx[i][k] == out_ctx[k]:
          pass  # This key has been already copied and
          # matches the output; no action needed
        elif key in out_ctx:
          remove_list.append(key)  # There is a discrepancy between
          # the ctxdata; set it to remove later
        else:
          out_ctx[key] = tmp_ctx[i][key]  # Copy the ctx data
        # end
      # end
    # end
    # Remove duplicates
    remove_list = list(dict.fromkeys(remove_list))
    # Remove the discrepancies
    for k in remove_list:
      out_ctx.pop(k)
    # end

    for i in range(num_out):
      grid_stack[-num_out + i].append(out_grid[i])
      value_stack[-num_out + i].append(out_values[i])
      ctx_stack[-num_out + i].append(out_ctx)
    # end
  # end
  return True


@click.command(
    help=f"Manipulate datasets using math expressions. Expressions are specified using Reverse Polish Notation (RPN).\n Supported operators are: {help_str[:-1]}"
)
@click.argument("chain", nargs=1, type=click.STRING)
@click.option("--tag", "-t", help="Tag for the result")
@click.option("--label", "-l", show_default=True, help="Custom label for the result")
@click.option("--all", "-a", is_flag=True, help="Ignore the status of a dataset")
@click.pass_context
def ev(ctx, **kwargs):
  verb_print(ctx, "Starting evaluate")
  data = ctx.obj["data"]

  grid_stack, value_stack, ctx_stack = [], [], []
  chain_split = kwargs["chain"].split(" ")
  chain_split = list(filter(None, chain_split))

  only_active = True
  if kwargs["all"]:
    only_active = False
  # end

  tags = list(data.tag_iterator(only_active=only_active))
  label = kwargs["label"]
  if label is None:
    label = kwargs["chain"]
  # end

  num_datasets_in_chain = 0
  out_data_id = ()
  for s in chain_split:
    is_data, data_id = _data(ctx, grid_stack, value_stack, ctx_stack, s, tags, only_active)
    if is_data and len(data_id) > 0 and data_id != out_data_id:
      num_datasets_in_chain += 1
      out_data_id = data_id
    # end
    if not is_data:
      is_command = _command(ctx, grid_stack, value_stack, ctx_stack, s)
    # end
    if not is_data and not is_command:
      ctx.fail(click.style(f"Evaluate input '{s:s}' represents neither data nor commad",
          fg="red"))
    # end
  # end

  if len(value_stack) == 0:
    ctx.fail(click.style("Evaluate stack is empty, there is nothing to return", fg="red"))
  elif len(value_stack) > 1:
    click.echo(
        click.style("WARNING: Length of the evaluate stack is bigger than 1, there is a posibility of unintended behavior",
            fg="yellow" ))
  # end
  if num_datasets_in_chain == 1 and kwargs["tag"] is None:
    cnt = 0
    tag = out_data_id[0]
    for out in ctx.obj["data"].iterator(tag=tag, select=out_data_id[1], only_active=only_active):
      out.push(grid_stack[-1][cnt], value_stack[-1][cnt])
      cnt += 1
    # end
  else:
    tag = out_data_id[0]
    if kwargs["tag"]:
      tag = kwargs["tag"]
    else:
      data.deactivate_all()
    # end
    for grid, values, data_ctx in zip(grid_stack[-1], value_stack[-1], ctx_stack[-1]):
      out = GData(tag=tag, # comp_grid=ctx.obj['compgrid'],
          label=label, ctx=data_ctx)
      out.push(grid, values)
      data.add(out)
    # end
  # end

  verb_print(ctx, "Finishing ev")
