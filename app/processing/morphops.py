# -*- coding: utf-8 -*-
"""
morphological operations
"""

import numpy as np
from skimage import morphology
from scipy.ndimage import binary_dilation, binary_erosion

import logging
logger = logging.getLogger(__name__)

def fill_holes_2D(mask,minsize=[]):
    out = np.copy(mask); out = out>0
    if not minsize:
        minsize = int(np.prod(mask.shape[:2])/2)
        rpt=True
        while rpt:
            rpt=False
            for z in range(mask.shape[-1]):
                out[:,:,z] = morphology.remove_small_holes(out[:,:,z], area_threshold=minsize, connectivity=1)
            if np.sum(out)==0:
                minsize = minsize-500
                rpt = True
    else:
        for z in range(mask.shape[-1]):
            out[:,:,z] = morphology.remove_small_holes(out[:,:,z], area_threshold=minsize, connectivity=1)
    return out

def dilation_2D(mask,se):
    out = mask>0
    for z in range(mask.shape[2]):
        out[:,:,z] = binary_dilation(out[:,:,z],se,iterations=1)
    return out

def erosion_2D(mask,se):
    out = mask>0
    for z in range(mask.shape[2]):
        out[:,:,z] = binary_erosion(out[:,:,z],se,iterations=1)
    return out

def opening_2D(mask,se):
    out = mask>0
    for z in range(mask.shape[2]):
        out[:,:,z] = morphology.binary_opening(out[:,:,z], se)
    return out

def closing_and_fill_2D(mask):
    se = morphology.square(4)
    mask = dilation_2D(mask,se)
    mask = fill_holes_2D(mask)
    mask = erosion_2D(mask,se)
    return mask

def closing_and_fill(mask):
    se = morphology.cube(6)
    mask = binary_dilation(mask,se,iterations=1)
    mask = fill_holes_2D(mask)
    mask = binary_erosion(mask,se,iterations=1)
    return mask

def closing_hull_2D(mask):
    mask = morphology.binary_closing(mask,morphology.cube(4))
    out = np.zeros(mask.shape)
    for z in range(mask.shape[2]):
        out[:,:,z] = morphology.convex_hull_image(mask[:,:,z])
    return out

def labelsize(mask, minsize=None, connectivity=1):
    labelmap, n = morphology.label(mask, connectivity=connectivity, background=0, return_num=True)
    lsizes = np.zeros(n+1, dtype='int')
    bIm = np.zeros(mask.shape, dtype='uint8')
    if not minsize==None:
        for label in range(1,n+1):
            aux = labelmap==(label)
            lsizes[label] = np.sum(aux)
            if lsizes[label]>=minsize:
                bIm[aux] = 1
        biglabel = np.argmax(lsizes)
    else:
        for label in range(1,n+1):
            lsizes[label] = np.sum(labelmap==label)
        biglabel = np.argmax(lsizes)
        bIm[labelmap==(biglabel)] = 1
    return bIm, lsizes, biglabel
