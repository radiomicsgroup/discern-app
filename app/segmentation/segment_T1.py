# -*- coding: utf-8 -*-
"""
T1 segmentation
"""

import logging
logger = logging.getLogger(__name__)

import os
import numpy as np
from skimage import morphology
import nrrd
from app.processing.preprocess import hist_perc
from app.processing.morphops import closing_and_fill, erosion_2D, labelsize


def and_brain_hems(p_hems, p_brain):
    logger.info('Combining Hems and Brain masks')
    brain, h = nrrd.read(p_brain)
    hems, _ = nrrd.read(p_hems)
    brain = np.bitwise_and(brain>0, hems>0)
    nrrd.write(p_brain, brain.astype('uint8'), h)
    return p_brain

def segment(im_t1_path, im_brain_mask, p_hems, output_path=[]):
    logger.info('Segmenting T1C')
    if not output_path:
        output_path1 = os.path.join(os.path.dirname(im_t1_path),'lesion-label.nrrd')
        output_path2 = os.path.join(os.path.dirname(im_t1_path),'lesion_filled-label.nrrd')
    else:
        output_path1 = os.path.join(output_path,'lesion-label.nrrd')
        output_path2 = os.path.join(output_path,'lesion_filled-label.nrrd')
    
    imt1,_ = nrrd.read(im_t1_path)
    
    mask_hems,_ = nrrd.read(p_hems)
    mask_brain,h = nrrd.read(im_brain_mask)
    mask_brain = np.bitwise_and(mask_brain==1, mask_hems>0)
    
    imt1sk = imt1.copy()
    imt1sk = imt1sk*(erosion_2D(mask_brain, morphology.square(4)))
    
    count, bins, hperc = hist_perc(imt1sk, imt1sk>0, 40)
    thhi = bins[np.argmax((count<np.mean(count))*(bins>bins[np.argmax(count)]))-1]
    thlesion=thhi
    
    lesion_mask = np.zeros(imt1sk.shape)
    lesion_mask[imt1sk>=thlesion] = 1
    
    lesionfinal = morphology.remove_small_objects(lesion_mask==1, min_size=10, connectivity=1)
    
    lesionfinal,_,_ = labelsize(lesionfinal)
    corefilled = closing_and_fill(lesionfinal)
    
    nrrd.write(output_path2, corefilled.astype('uint8'), h)
    nrrd.write(output_path1, lesionfinal.astype('uint8'), h)
    return output_path1, output_path2





