from os.path import isfile
from time import time
import sys

import click

def vlog(ctx, message):
    if ctx.obj['verbose']:
        elapsedTime = time() - ctx.obj['startTime']
        click.echo(click.style('[{:f}] {:s}'.format(elapsedTime, message),
                               fg='green'))

def pushChain(ctx, command, **kwargs):
    print(ctx.obj['savechainPath'])
    if ctx.obj['savechain'] == True:
        fh = open(ctx.obj['savechainPath'], 'a')
        s = '{:s}'.format(command)

        for key, value in kwargs.items():
            if sys.version_info[0] == 2:
                if type(value) == unicode:
                    value = str(value)
                #end
            #end
            if type(value) == str:
                value = "'" + value + "'"
            #end
            s = s + ', {:s}={}'.format(key, value)
        #end
        fh.write(s + '\n')
        fh.close()
    #end
#end

