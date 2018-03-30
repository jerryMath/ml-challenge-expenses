#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 18:57:43 2018

@author: jiayicheng
"""

import numpy as np
import operator

def classifier(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # use matrix format to calcalute the distences
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range (k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # pick the nearest one
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    
    return sortedClassCount[0][0]