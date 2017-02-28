import os
import json
import csv
from compiler.ast import flatten
import numpy as np


def _get_data(dir_path, path):
    """Private function to get the absolute path to the installed files."""
    cwd = os.path.abspath(os.path.dirname(__file__))
    joined = os.path.join(dir_path, path)
    out_dir = os.path.join(cwd, joined)
    return out_dir


def loadJSON(rfile):
    """Load json files"""

    return json.load(open(rfile, 'rb'))


def dlPull(data, key):
    """Pull out a list of values from a
    dictioanry based on key"""
    info = []
    for dataDict in data:
        for dat in dataDict:
            if not isinstance(dat[key], list):
                if dat[key] == u'C\xc3\xb4te d&#039;Ivoire' or dat[
                        key] == u'C\xf4te d&#039;Ivoire':
                    info.append('Ivory Coast')
                elif dat[key] == u'S\xe3o Tom\xe9 and Pr\xedncipe':
                    info.append('Sao Tome and Principe')
                else:
                    info.append(dat[key].encode('utf-8'))
            else:
                tmp = [x.encode('utf-8') for x in dat[key]]
                info.append(tmp)
    return info


def buildData(textFile, sYr, labelFile):

    # Make array out of text data
    path = _get_data('../../data', 'analysis')
    to_read = os.path.join(path, textFile)
    texts = loadJSON(to_read)
    tYr = []
    for slice in texts:
        tYr.extend([str(sYr)] * len(slice))
        sYr += 1
    texData = np.empty([len(tYr), 3], dtype=np.object)
    texData[:, 0] = [dlPull(texts, 'nameClean')[ii] + '_' + str(tYr[ii])
                     for ii in range(0, len(tYr))]
    texData[:, 1] = [' '.join(x) for x in dlPull(texts, 'dataClean')]
    texData[:, 2] = [len(x) for x in dlPull(texts, 'dataClean')]

    # Make array out of labeled data
    regime_path = _get_data('../../data', 'regimeData')
    labelFile = os.path.join(regime_path, labelFile)

    labList = []
    with open(labelFile, 'rU') as d:
        next(d)
        reader = csv.reader(d)
        for row in reader:
            cntryYr = row[1].lower() + '_' + str(row[2])
            labels = [row[col] for col in range(3, len(row))]
            labList.append(flatten([cntryYr, labels]))
    labData = np.array(labList)

    # Find intersections and differences
    inBoth = list(set(texData[:, 0]) & set(labData[:, 0]))
    niLab = list(set(texData[:, 0]) - set(labData[:, 0]))
    niTex = list(set(labData[:, 0]) - set(texData[:, 0]))

######
#This is where modifications have to happen depending on whether you're
#doing in-sample or true out-of-sample work. This is a difficult way to
#do this, but changing it requires changing the entirety of how the system
#works, from scrape to analysis.
######
    c = np.unique([x.split('_')[0] for x in inBoth])
    d = flatten([['{}_{}'.format(country, year) for country in c] for year in
                 tYr])
    tMatches = flatten(
        [[i for i, x in enumerate(texData[:, 0]) if x == cyr] for cyr in d])
#    tMatches = flatten(
#        [[i for i, x in enumerate(texData[:, 0]) if x == cyr] for cyr in inBoth])
#    lMatches = flatten(
#        [[i for i, x in enumerate(labData[:, 0]) if x == cyr] for cyr in inBoth])

    tlData = texData[tMatches, ]

#    tlData = np.hstack(
#        (texData[tMatches, ],
#         labData[lMatches, 1:labData.shape[1]]))
    return tlData
