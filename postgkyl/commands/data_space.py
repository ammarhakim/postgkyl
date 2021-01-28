import click
import numpy as np

class DataSpace(object):
    def __init__(self):
        self._datasetDict = {}
    #end

    #-----------------------------------------------------------------
    #-- Iterators ----------------------------------------------------
    def iterator(self, tag=None, enum=False, onlyActive=True):
        if tag:
            tags = tag.split(",")
        else:
            tags = list(self._datasetDict)
        #end
        for t in tags:
            try:
                for i, dat in enumerate(self._datasetDict[t]):
                    if (not onlyActive) or dat.getStatus(): # implication
                        if enum:
                            yield i, dat
                        else:
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
