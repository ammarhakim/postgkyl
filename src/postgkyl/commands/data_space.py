"""Postgkyl submodule to provide iterators in hte command line mode."""
from __future__ import annotations

import click
import numpy as np
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
  from postgkyl import GData
#end

class DataSpace(object):
  """Postgkyl class to store information about datasets and provide iterators in the command line mode."""

  def __init__(self):
    self._dataset_dict = {}

  # ---- Iterators ----
  def iterator(self, tag: str | None = None, enum: bool = False,
      only_active: bool = True, select: int | slice | str | None = None) -> Iterator[GData]:
    # Process 'select'
    if enum and select:
      click.echo(click.style("Error: 'select' and 'enum' cannot be selected simultaneously", fg="red"))
      quit()
    # end
    idx_sel = slice(None, None)
    if isinstance(select, int):
      idx_sel = [select]
    elif isinstance(select, slice):
      idx_sel = select
    elif isinstance(select, str):
      if ":" in select:
        lo = None
        up = None
        step = None
        s = select.split(":")
        if s[0]:
          lo = int(s[0])
        # end
        if s[1]:
          up = int(s[1])
        # end
        if len(s) > 2:
          step = int(s[2])
        # end
        idx_sel = slice(lo, up, step)
      else:
        idx_sel = list([int(s) for s in select.split(",")])
      # end
    # end

    if tag:
      tags = tag.split(",")
    else:
      tags = list(self._dataset_dict)
    # end
    for t in tags:
      try:
        if not select or isinstance(idx_sel, slice):
          for i, dat in enumerate(self._dataset_dict[t][idx_sel]):
            if (not only_active) or dat.get_status():  # implication
              if enum:
                yield i, dat
              else:
                yield dat
              # end
            # end
          # end
        else:  # isinstance(idx_sel, list)
          for i in idx_sel:
            dat = self._dataset_dict[t][i]
            if (not only_active) or dat.get_status():  # implication
              yield dat
            # end
          # end
        # end
      except KeyError as err:
        click.echo(click.style(f"ERROR: Failed to load the specified/default tag {err}", fg="red"))
        quit()
      except IndexError:
        click.echo(click.style("ERROR: Index out of the dataset range", fg="red"))
        quit()
      # end
    # end

  def tag_iterator(self, tag: str | None = None, only_active: bool = True) -> Iterator[str]:
    if tag:
      out = tag.split(",")
    elif only_active:
      out = []
      for t in self._dataset_dict:
        if True in (dat.get_status() for dat in self.iterator(t)):
          out.append(t)
        # end
      # end
    else:
      out = list(self._dataset_dict)
    # end
    return iter(out)

  # ---- Labels ----
  def set_unique_labels(self) -> None:
    num_comps = []
    names = []
    labels = []
    for dat in self.iterator():
      file_name = dat._file_name
      extension_len = len(file_name.split(".")[-1])
      file_name = file_name[: -(extension_len + 1)]
      # only remove the file extension but take into account
      # that the file name might start with '../'
      sp = file_name.split("_")
      names.append(sp)
      num_comps.append(int(len(sp)))
      labels.append("")
    # end
    max_elem = np.max(num_comps)
    idx_max = np.argmax(num_comps)
    for i in range(max_elem):
      include = False
      reference = names[idx_max][i]
      for nm in names:
        if i < len(nm) and nm[i] != reference:
          include = True
        # end
      # end
      if include:
        for idx, nm in enumerate(names):
          if i < len(nm):
            if labels[idx] == "":
              labels[idx] += nm[i]
            else:
              labels[idx] += f"_{nm[i]:s}"
            # end
          # end
        # end
      # end
    # end
    cnt = 0
    for dat in self.iterator():
      dat.set_label(labels[cnt])
      cnt += 1
    # end

  # ---- Adding datasets ----
  def add(self, data: GData) -> None:
    tag_nm = data.get_tag()
    if tag_nm in self._dataset_dict:
      self._dataset_dict[tag_nm].append(data)
    else:
      self._dataset_dict[tag_nm] = [data]
    # end

  # ---- Staus control ----
  def activate_all(self, tag: str | None = None) -> None:
    for dat in self.iterator(tag=tag, only_active=False):
      dat.deactivate()
    # end

  # end
  def deactivate_all(self, tag: str | None = None) -> None:
    for dat in self.iterator(tag=tag, only_active=False):
      dat.deactivate()
    # end

  # ---- Utilities ----
  def get_dataset(self, idx: int, tag: str = "default") -> GData:
    return self._dataset_dict[tag][idx]


  def get_num_datasets(self, tag: str | None = None, only_active: bool = True):
    num_sets = 0
    for dat in self.iterator(tag=tag, only_active=only_active):
      num_sets += 1
    # end
    return num_sets

  def clean(self):
    self._dataset_dict = {}