from time import time
import click

def verb_print(ctx: click.core.Context, message: str) -> None:
  if ctx.obj["verbose"]:
    elapsed_time = time() - ctx.obj["start_time"]
    click.echo(click.style(f"[{elapsed_time:f}] {message:s}", fg="green"))
  # end
