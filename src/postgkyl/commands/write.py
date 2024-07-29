import click
import shutil

from postgkyl.utils import verb_print


@click.command()
@click.option("--use", "-u", help="Specify a 'tag' to apply to (default all tags).")
@click.option("-f", "--filename", type=click.STRING, prompt=True, help="Output file name.")
@click.option("-m", "--mode", type=click.Choice(["gkyl", "bp", "txt", "npy"]), default="gkyl",
    help="Output file mode. One of `gkyl` (binary, default), `bp` (ADIOS BP file), `txt` (ASCII text file), or `npy` (NumPy binary file).")
@click.option("-s", "--single", is_flag=True, help="Write all dataset into one file")
@click.pass_context
def write(ctx, **kwargs):
  """Write active dataset to a file.

  The output file format can be set with ``--format``, and is Gkeyll's .gkyl by default.
  Files saved as .gkyl or .bp can be later loaded back into pgkyl to further manipulate
  or plot.
  """
  verb_print(ctx, "Starting write")
  data = ctx.obj["data"]

  var_name = None
  append = False
  cleaning = True
  fn = kwargs["filename"]
  mode = kwargs["mode"]
  if len(fn.split(".")) > 1:
    mode = str(fn.split(".")[-1])
    fn = str(fn.split(".")[0])
  # end

  num_files = data.get_num_datasets(tag=kwargs["use"])
  for i, dat in data.iterator(tag=kwargs["use"], enum=True):
    out_name = f"{fn:s}.{mode:s}"
    if kwargs["single"]:
      var_name = f"{dat.get_tag():s}_{i:d}"
      cleaning = False
    else:
      if num_files > 1:
        out_name = f"{fn:s}_{i:d}.{mode:s}"
      # end
    # end

    dat.write(out_name=out_name, mode=mode, append=append, var_name=var_name, cleaning=cleaning)

    if kwargs["single"]:
      append = True
    # end
  # end

  # Cleaning
  if not cleaning:
    shutil.move(f"{fn:s}.{mode:s}.dir/{fn:s}.{mode:s}.0", f"{fn:s}.{mode:s}")
    shutil.rmtree(f"{fn:s}.{mode:s}.dir")
  # end
  verb_print(ctx, "Finishing write")
