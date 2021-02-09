import click
import re
from glob import glob

from postgkyl.commands.util import vlog

@click.command()
@click.option('--extension', '-e', type=click.STRING,
              default='bp', show_default=True,
              help='Output file extension')
@click.pass_context
def listoutputs(ctx, **kwargs):
    """List Gkeyll filename stems in the current directory
    """
    vlog(ctx, 'Starting listoutputs')
        
    files = glob('*.{:s}'.format(kwargs['extension']))
    unique = []
    for fn in files:
        # remove extension
        s = fn[:-(len(kwargs['extension'])+1)]
        # strip "restart"
        if s.endswith('_restart'):
            s = s[:-8]
        #end
        #strip digits
        s = re.sub(r'_\d+$', '', s)
        if s not in unique:
            unique.append(s)
        #end
    #end
    for s in sorted(unique):
        click.echo(s)
    #end
    
    vlog(ctx, 'Finishing listoutputs')
#end
