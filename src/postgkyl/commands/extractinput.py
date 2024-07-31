import base64
import click

from postgkyl.utils import verb_print


@click.command()
@click.option("--use", "-u", help="Specify a 'tag' to apply to (default all tags).")
@click.pass_context
def extractinput(ctx, **kwargs):
  """Extract embedded input file from compatible BP files"""
  verb_print(ctx, "Starting ")
  data = ctx.obj["data"]

  for dat in data.iterator(kwargs["use"]):
    enc_inp = dat.get_input_file()
    if enc_inp:
      inpfile = base64.decodebytes(enc_inp.encode("utf-8")).decode("utf-8")
      click.echo(inpfile)
    else:
      click.echo("No embedded input file!")
    # end
  # end
  verb_print(ctx, "Finishing extractinput")
