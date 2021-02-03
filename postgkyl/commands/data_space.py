import click
import numpy as np

class DataSpace(object):
    def __init__(self):
        self._datasetDict = {}
    #end

    #-----------------------------------------------------------------
    #-- Iterators ----------------------------------------------------

    def _getIterableIdx(idx, length):
        def _int(i, length):
            i = int(i)
            if i >= 0:
                return i
            else:
                return int(length + i)
            #end
        #end
    
        if idx is None:
            return range(length)
        elif ',' in idx:
            s = idx.split(',')
            return [_int(i, length) for i in s]
        elif ':' in idx:
            s = idx.split(':')
            si = [0, length, 1]
            if s[0]:
                si[0] = _int(s[0], length)
            if s[1]:
                si[1] = _int(s[1], length)
            if len(s) > 2:
                si[2] = int(s[2])
            return range(si[0], si[1], si[2])
        else:
            return [_int(idx, length)]
        #end
    #end
    
    def iterator(self, tag=None, enum=False, onlyActive=True, select=None):
        # Process 'select'
        if enum and select:
            click.echo(click.style("Error: 'select' and 'enum' cannot be selected simultaneously", fg='red'))
            quit()
        #end
        idxSel = slice(None, None)
        if isinstance(select, int):
            idxSel = [select]
        elif isinstance(select, slice):
            idxSel = select
        elif isinstance(select, str):
            if ':' in select:
                lo = None
                up = None
                step = None
                s = select.split(':')
                if s[0]:
                    lo = int(s[0])
                #end
                if s[1]:
                    up = int(s[1])
                #end
                if len(s) > 2:
                    step = int(s[2])
                #end
                idxSel = slice(lo, up, step)
            else:
                idxSel = list([int(s) for s in select.split(',')])
        #end

        if tag:
            tags = tag.split(",")
        else:
            tags = list(self._datasetDict)
        #end
        for t in tags:
            try:
                if not select or isinstance(idxSel, slice):
                    for i, dat in enumerate(self._datasetDict[t][idxSel]):
                        if (not onlyActive) or dat.getStatus(): # implication
                            if enum:
                                yield i, dat 
                            else:
                                yield dat
                            #end
                        #end    
                    #end
                else: #isinstance(idxSel, list)
                    for i in idxSel:
                        dat = self._datasetDict[t][i]
                        if (not onlyActive) or dat.getStatus(): # implication
                            yield dat
                        #end    
                    #end
                #end
            except KeyError as err:
                click.echo(click.style("ERROR: Failed to load the specified/default tag {0}".format(err),
                                       fg='red'))
                quit()
            #end
        #end
    #end
    
    def tagIterator(self, tag=None, onlyActive=True):
        if tag:
            out = tag.split(',')
        elif onlyActive:
            out = []
            for t in self._datasetDict:
                if True in (dat.getStatus() for dat in self.iterator(t)):
                    out.append(t)
                #end
            #end
        else:
            out = list(self._datasetDict)
        #end
        return iter(out)
    #end
        

    #-----------------------------------------------------------------
    #-- Labels -------------------------------------------------------
    def setUniqueLabels(self):
        numComps = []
        names = []
        labels = []
        for dat in self.iterator():
            fileName = dat.fileName
            extensionLen = len(fileName.split('.')[-1])
            fileName = fileName[:-(extensionLen+1)]
            # only remove the file extension but take into account
            # that the file name might start with '../'
            sp = fileName.split('_')
            names.append(sp)
            numComps.append(int(len(sp)))
            labels.append("")
        #end
        maxElem = np.max(numComps)
        idxMax = np.argmax(numComps)
        for i in range(maxElem): 
            include = False
            reference = names[idxMax][i]
            for nm in names:
                if i < len(nm) and nm[i] != reference:
                    include = True
                #end
            #end
            if include:
                for idx, nm in enumerate(names):
                    if i < len(nm):
                        if labels[idx] == "":
                            labels[idx] += nm[i]
                        else:
                            labels[idx] += '_{:s}'.format(nm[i])
                        #end
                    #end
                #end
            #end
        #end
        cnt = 0
        for dat in self.iterator():
            dat.setLabel(labels[cnt])
            cnt += 1
        #end
    #end
    
    #-----------------------------------------------------------------
    #-- Adding datasets ----------------------------------------------
    def add(self, data):
        tagNm = data.getTag()
        if tagNm in self._datasetDict:
            self._datasetDict[tagNm].append(data)
        else:
            self._datasetDict[tagNm] = [data]
        #end
    #end

    #-----------------------------------------------------------------
    #-- Staus control ------------------------------------------------
    def activateAll(self, tag=None):
        for dat in self.iterator(tag=tag, onlyActive=False):
            dat.deactivate()
        #end
    #end
    def deactivateAll(self, tag=None):
        for dat in self.iterator(tag=tag, onlyActive=False):
            dat.deactivate()
        #end
    #end
    
    #-----------------------------------------------------------------
    #-- Stuff :-P ----------------------------------------------------
    def getDataset(self, tag, idx):
        return self._datasetDict[tag][idx]
    #end
        
    def getNumDatasets(self, tag=None, onlyActive=True):
        numSets = 0
        for dat in self.iterator(tag=tag, onlyActive=onlyActive):
            numSets += 1
        #end
        return numSets
    #end
#end
