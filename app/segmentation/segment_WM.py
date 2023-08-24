# -*- coding: utf-8 -*-
"""
WM segmentation
"""

import os
import numpy as np
import nrrd
from app.processing.preprocess import hist_perc
from app.processing.morphops import fill_holes_2D, labelsize
from skimage import morphology

import logging
logger = logging.getLogger(__name__)


def segment(im_t1_path, im_brain_mask, im_hems, im_lesion_t1, output_path=[]):
    logger.info('Segmenting WM')
    if not output_path:
        output_path = os.path.dirname(im_t1_path)
    outfolder = os.path.join(output_path)
    mask_brain, headperf = nrrd.read(im_brain_mask)
    mask_brain[mask_brain>0] = 1
    
    imt1,ht1 = nrrd.read(im_t1_path)
    imt1sk = imt1.copy()
    imt1sk[mask_brain==0] = 0
    
    brainlabels, hbl = nrrd.read(im_hems)
    
    lesionfinal,_ = nrrd.read(im_lesion_t1)
    
    thems = brainlabels[lesionfinal>0]
    rhem = np.sum(thems==24) + np.sum(thems==21)
    lhem = np.sum(thems==25) + np.sum(thems==22)
    if rhem<lhem:
        wmlb=21
        cleanhem = np.bitwise_or(brainlabels==24, brainlabels==21)
    else:
        wmlb=22
        cleanhem = np.bitwise_or(brainlabels==25, brainlabels==22)
    
    cleanhem[mask_brain==0] = 0
    cleanhem[lesionfinal!=0] = 0
    nrrd.write(os.path.join(outfolder,'HEM-label.nrrd'), cleanhem.astype('uint8'), headperf)
    
    
    count2, bins2, hp2 = hist_perc(imt1sk, cleanhem, 40)
    thwm = bins2[np.argmax(count2)]
    if hp2[(bins2>=thwm)][0]>0.8:
        thwm = bins2[hp2>=0.5][0]
    
    wmmask = np.copy(cleanhem)
    wmmask = fill_holes_2D(wmmask, 10000)
    if any(bins2[(hp2>=0.95)]):
        wmmask[imt1sk>(bins2[(hp2>=0.95)][0])] = 0
    wmmask[imt1sk<thwm] = 0
    wmmask,_,_ = labelsize(wmmask)
    if not np.sum(wmmask)>=np.sum(cleanhem)/5:
        logging.debug('WM segment may be wrong (too small)')
    else:
        wmmask = morphology.binary_opening(wmmask,morphology.ball(1))
    wmsz = np.sum(wmmask)
    if wmsz<500 and np.sum(brainlabels==wmlb)>wmsz:
        logging.info('Using Slicer WM segmentation')
        wmmask = np.zeros(cleanhem.shape)
        wmmask[brainlabels==wmlb] = 1
    
    nrrd.write(os.path.join(outfolder,'WM-label.nrrd'), wmmask.astype('uint8'), headperf)
    return os.path.join(outfolder,'WM-label.nrrd')

