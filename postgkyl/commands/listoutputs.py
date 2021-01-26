import click

from glob import glob

from postgkyl.commands.util import vlog

@click.command()
@click.option('--extension', '-e', type=click.STRING,
              default='bp', show_default=True,
              help='Output file extension')
@click.pass_context
def listoutputs(ctx, **kwargs):
    vlog(ctx, 'Starting listoutputs')
        
    files = glob('*[0-9].{:s}'.format(kwargs['extension']))
    unique = []
    for fn in files:
        # remove the frame number
        ext = fn.split('_')[-1]
        # get the stem
        s = fn[:-(len(ext)+1)]
        if s not in unique:
            unique.append(s)
        #end
    #end
    for s in sorted(unique):
        click.echo(s)
    #end
    
    vlog(ctx, 'Finishing listoutputs')
#end
