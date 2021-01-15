import numpy as np

class DataSpace(object):
    def __init__(self):
        self._datasetDict = {}
    #end

    #-----------------------------------------------------------------
    # Iterators  
    def iterator(self, tag=None):
        if tag:
            for i in range(len(self._datasetDict[tag])):
                yield self._datasetDict[tag][i]
            #end
        else:
            for t in self._datasetDict:
                for i in range(len(self._datasetDict[t])):
                    yield self._datasetDict[t][i]
                #end
            #end
        #end
    #end

    #-----------------------------------------------------------------
    # Labels
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
        for idx, dat in enumerate(self.iterator()):
            dat.setLabel(labels[idx])
        #end
    #end
    
    #-----------------------------------------------------------------
    # Adding datasets
    def add(self, data):
        tagNm = data.getTag()
        if tagNm in self._datasetDict:
            self._datasetDict[tagNm].append(data)
        else:
            self._datasetDict[tagNm] = [data]
        #end
    #end
    
    #-----------------------------------------------------------------
    # Stuff
    def getNumDatasets(self, tag=None):
        numDatasets = 0
        if tag:
            numDatasets = len(self._datasetDict[tag])
        else:
            for t in self._datasetDict:
                numDatasets += len(self._datasetDict[t])
            #end
        #end
        return numDatasets
    #end
#end
