import click

from postgkyl.data import GData
import postgkyl.tools
from postgkyl.utils import verb_print


@click.command()
@click.option("--distribution", "-f", type=click.STRING, prompt=True,
    help="Specify the PKPM distribution function dataset.")
@click.option("--tm", type=click.STRING, prompt=True, help="Specify the PKPM vars dataset.")
@click.option("--tag", "-t", help="Optional tag for the resulting array")
@click.option("--label", "-l", help="Custom label for the result")
@click.pass_context
def laguerrecompose(ctx, **kwargs):
  """Compose PKPM Laguerre coefficients together."""
  verb_print(ctx, "Starting laguerrecompose")
  data = ctx.obj["data"]

  for f, tm in zip(data.iterator(kwargs["distribution"]), data.iterator(kwargs["tm"])):
    if kwargs["tag"]:
      out = GData(tag=kwargs["tag"], label=kwargs["label"],
          comp_grid=ctx.obj["compgrid"], ctx=f.ctx)
      postgkyl.tools.laguerre_compose(f, tm, out)
    else:
      postgkyl.tools.laguerre_compose(f, tm, f)
    # end
  # end
  verb_print(ctx, "Finishing laguerrecompose")
