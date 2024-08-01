import click

from postgkyl.utils import load_style, verb_print


@click.command()
@click.option("--file", "-f", help="Sets Maplotlib rcParams style file.")
@click.option("--set", "-s", multiple=True, help="Sets individual rcParam(s) as 'key:value'.")
@click.option("--print", "-p", is_flag=True, help="Prints the current rcParams.")
@click.pass_context
def style(ctx, **kwargs):
  """Probe and control the Matplotlib plotting style.

  The list of rcParams is available
  here:\nhttps://matplotlib.org/stable/api/matplotlib_configuration_api.html"""
  verb_print(ctx, "Starting 'style' command")

  if kwargs["file"]:
    load_style(ctx, kwargs["file"])
  # end

  for param in kwargs["set"]:
    param_split = param.split(":")
    key = param_split[0].strip()
    value = param[len(param_split[0]) + 1 :].strip()
    ctx.obj["rcParams"][key] = value
  # end

  if kwargs["print"]:
    for key in ctx.obj["rcParams"]:
      print(f"{key:s} : {ctx.obj['rcParams'][key]}")
    # end
  # end

  verb_print(ctx, "Finishing 'style' command")
