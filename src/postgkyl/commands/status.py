import click

from postgkyl.utils import verb_print


@click.command()
@click.option("--tag", "-t", type=click.STRING, help="Tag(s) to apply to (comma-separated).")
@click.option("--index", "-i", type=click.STRING,
    help="Dataset indices (e.g., '1', '0,2,5', or '1:6:2').")
@click.option("--focused", "-f", is_flag=True, help="Leave unspecified datasets untouched.")
@click.pass_context
def activate(ctx, **kwargs):
  """Select datasets(s) to pass further down the command chain.

  Datasets are indexed starting 0. Multiple datasets can be selected using a comma
  separated list or a range specifier. Unless '--focused' is selected, all unselected
  datasets will be deactivated.

  '--tag' and '--index' allow to specify tags and indices. The not specified, 'activate'
  applies to all. Both parameters support comma-separated values. '--index' also
  supports slices following the Python conventions, e.g., '3:7' or ':-5:2'.

  'info' command (especially with the '-ac' flags) can be helpful when
  activating/deactivating multiple datasets.
  """
  verb_print(ctx, "Starting activate")
  data = ctx.obj["data"]

  if not kwargs["focused"]:
    data.deactivate_all()
  # end

  for dat in data.iterator(tag=kwargs["tag"], only_active=False, select=kwargs["index"]):
    dat.activate()
  # end

  verb_print(ctx, "Finishing activate")


@click.command()
@click.option("--tag", "-t", type=click.STRING, help="Tag(s) to apply to (comma-separated).")
@click.option("--index", "-i", type=click.STRING,
    help="Dataset indices (e.g., '1', '0,2,5', or '1:6:2').")
@click.option("--focused", "-f", is_flag=True, help="Leave unspecified datasets untouched.")
@click.pass_context
def deactivate(ctx, **kwargs):
  """Select datasets(s) to pass further down the command chain.

  Datasets are indexed starting 0. Multiple datasets can be selected using a comma
  separated list or a range specifier. Unless '--focused' is selected, all unselected
  datasets will be activated.

  '--tag' and '--index' allow to specify tags and indices. The not specified,
  'deactivate' applies to all. Both parameters support comma-separated values. '--index'
  also supports slices following the Python conventions, e.g., '3:7' or ':-5:2'.

  'info' command (especially with the '-ac' flags) can be helpful when
  activating/deactivating multiple datasets.
  """
  verb_print(ctx, "Starting deactivate")
  data = ctx.obj["data"]

  if kwargs["focused"]:
    data.activate_all()
  # end

  for dat in data.iterator(tag=kwargs["tag"], only_active=False, select=kwargs["index"]):
    dat.deactivate()
  # end

  verb_print(ctx, "Finishing deactivate")
