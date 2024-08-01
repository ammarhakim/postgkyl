import click

from postgkyl.utils import verb_print


@click.command(help="Print info of active datasets.")
@click.option("-u", "--use", help="Specify a 'tag' to apply to (default all tags).")
@click.option("-c", "--compact", is_flag=True, help="Show in compact mode.")
@click.option("-a", "--allsets", is_flag=True, help="All data sets.")
@click.pass_context
def info(ctx, **kwargs):
  verb_print(ctx, "Starting info")
  data = ctx.obj["data"]
  if kwargs["allsets"]:
    only_active = False
  else:
    only_active = True
  # end

  for i, dat in data.iterator(kwargs["use"], enum=True, only_active=only_active):
    if dat.get_status():
      color = "green"
      bold = True
    else:
      color = None
      bold = False
    # end
    click.echo(
        click.style(f"{dat.get_label():s}{' ' if dat.get_label() else '':s}({dat.get_tag():s}#{i:d})",
            fg=color, bold=bold)
    )
    if not kwargs["compact"]:
      click.echo(dat.info() + "\n")
    # end
  # end

  verb_print(ctx, "Finishing info")
