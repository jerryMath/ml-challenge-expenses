#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 18:49:45 2018

@author: jiayicheng
"""
import numpy as np

def dis(vec1, vec2):
    """ Calculates the euclidean distance between 2 lists of coordinates. """
    return np.sqrt(np.sum((vec1 - vec2)**2))