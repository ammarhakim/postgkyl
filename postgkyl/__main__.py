
import click
import numpy

import postgkyl.commands as cmd
from postgkyl.data.load import GData
from postgkyl.data.interp import GInterpZeroOrder

@click.group(chain=True)
@click.option('--filename', '-f', multiple=True,
              help='Specify one or more file(s) to work with.')
@click.pass_context
def cli(ctx, filename):
    if filename == ():
        click.echo('No data file given. Specify file(s) with \'-f\'')
        ctx.exit()
    
    ctx.obj['files'] = filename
    numFiles = len(filename)
    ctx.obj['numFiles'] = numFiles
    ctx.obj['data'] = []
    ctx.obj['numComps'] = []
    ctx.obj['coords'] = []
    ctx.obj['values'] = []
    for f in range(numFiles):
        ctx.obj['data'].append(GData(filename[f]))
        numComps = ctx.obj['data'][f].q.shape[-1]
        ctx.obj['numComps'].append(numComps)
        
        dg = GInterpZeroOrder(ctx.obj['data'][f])
        coords, values = dg.project(0)
        if numComps > 1:
            values = numpy.extend_dims(values,
                                       axis=ctx.obj['data'][f].numDims)
            for c in numpy.arange(numComps-1)+1:
                coords, v = dg.project(c)
                v = numpy.extend_dims(v, axis=ctx.obj['data'][f].numDims)
                values = numpy.append(values, v,
                                      axis=ctx.obj['data'][f].numDims)
        ctx.obj['coords'].append(coords)
        ctx.obj['values'].append(values)
        print(ctx.obj['values'].shape)

cli.add_command(cmd.info.info)
cli.add_command(cmd.output.plot)
cli.add_command(cmd.project.project)
cli.add_command(cmd.transform.mult)
cli.add_command(cmd.transform.norm)

if __name__ == '__main__':
    cli(obj={})
