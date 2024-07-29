from glob import glob
import click
import re

from postgkyl.utils import verb_print


@click.command()
@click.option("--extensions", "-e", type=click.STRING,  default="bp,gkyl",
    show_default=True, help="Output file extension(s)")
@click.pass_context
def listoutputs(ctx, **kwargs):
  """List Gkeyll filename stems in the current directory."""
  verb_print(ctx, "Starting listoutputs")

  extensions = kwargs["extensions"].split(",")
  for ext in extensions:
    files = glob(f"*.{ext:s}")
    unique = []
    for fn in files:
      # remove extension
      s = fn[: -(len(ext) + 1)]
      # strip "restart"
      if s.endswith("_restart"):
        s = s[:-8]
      # end
      # strip digits
      s = re.sub(r"_\d+$", "", s)
      if s not in unique:
        unique.append(s)
      # end
    # end
    if len(unique) > 0:
      click.echo(f"{ext:s}:")
    # end
    for s in sorted(unique):
      click.echo(f"- {s:s}")
    # end
  # end
  verb_print(ctx, "Finishing listoutputs")
