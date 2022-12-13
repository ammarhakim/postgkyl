import click

@click.command(help='Run the saved command chain')
@click.option('--filename', '-f',
              help="Specify file with stored chain (default 'pgkylchain.dat')")
@click.pass_context
def runchain(ctx, filename):
  if filename is None:
    fn = ctx.obj['savechainPath']
  else:
    home = os.path.expanduser('~')
    fn = home + '/.pgkyl/' + filename
  #end
  if os.path.isfile(fn):
    fh = open(fn, 'r')
    for line in fh.readlines():
      eval('ctx.invoke(cmd.{:s})'.format(line))
    #end
    fh.close()
  else:
    raise NameError("File with stored chain ({:s}) does not exist".
                    format(filename))
  #end
#end
