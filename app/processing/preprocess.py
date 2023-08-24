# -*- coding: utf-8 -*-
"""
preprocess of T1w-CE
"""

import os
from app.utils import cast_to_type, end_execution, get_exitcode_stdout_stderr, wrap_string
import numpy as np
import nrrd
from skimage import morphology, filters

import logging
logger = logging.getLogger(__name__)

def mypdf(data):
    pdf = np.zeros(len(data))
    for h,val in enumerate(data):
        if h>0:
            pdf[h] = pdf[h-1]+val
        else:
            pdf[h] = val
    return pdf

def hist_perc(im, im_mask=np.zeros(1), nbin=100):
    if im_mask.any():
        count, bins = np.histogram(im[im_mask], nbin, density=False)
    else:
        count, bins = np.histogram(im, nbin, density=False)
    bins = bins[1:]
    count = count/np.sum(count)
    hperc = mypdf(count)
    return [count, bins, hperc]


def correct_even_odd(imt1):
    imt1[imt1<0] = 0
    thbg = filters.threshold_li(imt1)
    
    c,b,_ = hist_perc(imt1, imt1>=thbg, nbin=40)
    thhi = b[np.argmax((c<np.mean(c))*(b>b[np.argmax(c)]))-1]
    thlo = b[np.argmax((c<np.mean(c))*(b<b[np.argmax(c)]))]
    if not thlo.any():
        thlo = b[np.argmin(c[b<b[np.argmax(c)]])]
    im_mask = (imt1>thlo)*(imt1<thhi)
  
    ime = imt1[:,:,::2]
    imo = imt1[:,:,1::2]
    
    ce,be,_ = hist_perc(ime, im_mask[:,:,::2], nbin=40)
    co,bo,_ = hist_perc(imo, im_mask[:,:,1::2], nbin=40)
    
    xcor = np.correlate(ce, co, mode = 'same')
    lag = int(np.argmax(xcor)-xcor.size/2)
    new_img = imt1.copy()
    if lag!=0:
        binratio = 1/(bo[np.argmax(co)]/be[np.argmax(co)+lag])
        new_img = imt1.copy()
        new_img[:,:,1::2] = imo*binratio
    return new_img


def sk_strip_T1(imt1):
    se2 = np.zeros((9,9,9))
    se2[:,:,4] = 1
    se = morphology.cube(5)
    th_fore = filters.threshold_li(imt1)
    
    hist_all = np.histogram(imt1,20, density=False)
    id_bin = np.argmin(np.absolute(hist_all[1]-th_fore))
    hist_fore_bins = hist_all[1][(id_bin+1)::1]
    hist_fore_count = hist_all[0][id_bin::1]
    dx = np.diff(hist_fore_count)
    id_th_hi = np.argmin(dx)+1
    th_hi = hist_fore_bins[id_th_hi]
    id_hi = imt1>th_hi
    
    mask_th = np.ones(imt1.shape, dtype=bool)
    id_bg = imt1<th_fore
    mask_th[id_bg] = 0
    mask_th[id_hi] = 0
    
    mask_er = morphology.binary_erosion(mask_th, se)
    total_size = sum(sum(sum(mask_er)))
    mask_clean = morphology.remove_small_objects(mask_er,total_size/2,1)
    mask_di = morphology.binary_dilation(mask_clean, se)
    
    mask_cl = morphology.binary_closing(mask_di, se2)
    if sum(sum(sum(mask_cl)))>0.5*np.prod(mask_clean.shape):
        print('Closing result may be wrong! Check SE size\n')
    mask_fill = morphology.remove_small_holes(mask_cl,100000,2)
    
    return [mask_fill, th_fore]


def correct_T1(input_path, output_path, slicer_exe):
    logger.info('correcting T1C')
    if not output_path:
        output_path = os.path.join(os.path.dirname(input_path),'T1_corrected')
    os.makedirs(output_path, exist_ok=True)
    
    if not os.path.exists(output_path): os.makedirs(output_path)
    
    imt1, ht1 = nrrd.read(os.path.join(input_path))
    
    logger.debug('correcting T1C inter-leave')
    im1 = correct_even_odd(imt1)
    nrrd.write(os.path.join(output_path,'T1_eo.nrrd'), im1, ht1)
    
    logger.debug('correcting T1C bias field')
    
    cli_exe = slicer_exe + ' --launch N4ITKBiasFieldCorrection'
    files_io = [os.path.join(output_path,'T1_eo.nrrd'), os.path.join(output_path,'T1_n4.nrrd')]
    files_io = [f.replace('\\','/') for f in files_io]
    files_io = [wrap_string(f,'"') for f in files_io]
    all_io = ' '.join(files_io)
    params_str = '--meshresolution 1,1,1 --splinedistance 0 --bffwhm 0 --iterations 50,40,30 --convergencethreshold 0.0001 --bsplineorder 3 --shrinkfactor 4 --wienerfilternoise 0 --nhistogrambins 0'
    call_str = " ".join([cli_exe, params_str, all_io])
    exitcode, out, err = get_exitcode_stdout_stderr(call_str)
    if exitcode:
        logger.error('N4 bias correction terminated with error code: %s', err.decode())
        end_execution()
    i, h = nrrd.read(os.path.join(output_path,'T1_n4.nrrd'))
    nrrd.write(os.path.join(output_path,'T1_n4.nrrd'), cast_to_type(i, im1.dtype.name), h)
    return os.path.join(output_path,'T1_n4.nrrd'), os.path.join(output_path,'T1_eo.nrrd')
    
    