from time import time
from cycler import cycler
import numpy as np
import click


def verb_print(ctx, message):
  if ctx.obj["verbose"]:
    elapsedTime = time() - ctx.obj["start_time"]
    click.echo(click.style("[{:f}] {:s}".format(elapsedTime, message), fg="green"))
  # end


def load_style(ctx, fn):
  fh = open(fn, "r")
  for line in fh.readlines():
    key = line.split(":")[0]
    key_len = int(len(key))
    key = key.strip()
    value = line[(key_len + 1) :].strip()
    if value[:6] == "cycler":
      arg = eval(value[16:-1])
      value = cycler(color=arg)
    # end
    ctx.obj["rcParams"][key] = value
  # end
  fh.close()


def _find_nearest_index(array, value):
  if array is None:
    raise TypeError(
        "The index value is float but the 'array' from which to select the neares value is not specified."
    )
  # end
  idx = np.searchsorted(array, value)
  if idx == len(array):
    return int(idx - 2)
  elif idx > 0:
    return int(idx - 1)
  else:
    return int(idx)
  # end


def _find_cell_index(array, value):
  if array is None:
    raise TypeError(
        "The index value is float but the 'array' from which to select the neares value is not specified."
    )
  # end
  idx = np.searchsorted(array, value)
  return int(idx)


def _string_to_index(value, array=None, nodal=False):
  if isinstance(value, str):
    if value.isdigit():
      return int(value)
    else:
      if nodal:
        return _find_cell_index(array, float(value))
      else:
        return _find_nearest_index(array, float(value))
      # end
    # end
  else:
    raise TypeError("Value is not string")
  # end


def idx_parser(value, array=None, nodal=False):
  idx = None
  if isinstance(value, int):
    idx = value
  elif isinstance(value, float):
    if nodal:
      idx = _find_cell_index(array, value)
    else:
      idx = _find_nearest_index(array, value)
    # end
  else:
    if isinstance(value, str):
      if len(value.split(",")) > 1:
        idxs = value.split(",")
        idx = tuple([_string_to_index(i, array, nodal) for i in idxs])
      elif len(value.split(":")) == 2:
        idxs = value.split(":")
        if idxs[0] == "":
          idxs[0] = str(0)
        # end
        if idxs[1] == "":
          idxs[1] = str(len(array))
        # end
        try:
          if int(idxs[1]) < 0:
            idxs[1] = str(len(array) + int(idxs[1]) + 1)
          # end
        except ValueError:
          pass
        idx = slice(
            _string_to_index(idxs[0], array, nodal),
            _string_to_index(idxs[1], array, nodal),
        )
      else:
        idx = _string_to_index(value, array, nodal)
      # end
    # end
  # end

  return idx
