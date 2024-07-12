from time import time
import click

def verb_print(ctx, message):
  if ctx.obj["verbose"]:
    elapsedTime = time() - ctx.obj["start_time"]
    click.echo(click.style("[{:f}] {:s}".format(elapsedTime, message), fg="green"))
  # end