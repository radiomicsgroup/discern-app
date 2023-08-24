#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
input/output utilities
"""

import logging
from turtle import color
logger = logging.getLogger(__name__)

import os
import nrrd
from app.utils import cast_to_type
import SimpleITK as sitk
import numpy as np


def split_filename(filepath):
    """ split a filepath into the full path, filename, and extension (works with .nii.gz) """
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext

def open_nrrd(filepath):
    """ open a nrrd file and return the object """
    image = os.path.abspath(os.path.expanduser(filepath))
    obj = nrrd.read(image)
    return obj

def save_nrrd(obj, outfile, dtype=[]):
    """ save a nrrd object = [image, header] """
    if dtype:
        newarray = cast_to_type(obj[0], dtype)
        nrrd.write(outfile, newarray, obj[1])
    else:
        nrrd.write(outfile, obj[0], obj[1])

def write_image_sitk(outputImageFileName, image, isUInt=False):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(outputImageFileName)
    if isUInt:
        if not image.GetPixelID()==3:
            sitk.Cast(image, sitk.sitkUInt16)
    writer.Execute(image)

def read_image_sitk(inputImageFileName, isUInt=False, isMask=False):
    reader = sitk.ImageFileReader()
    base, ext = os.path.splitext(inputImageFileName)
    if ext=='nrrd' or ext=='nhdr':
        reader.SetImageIO("NrrdImageIO")
    elif ext=='nii' or (ext=='gz' and 'nii' in base):
        reader.SetImageIO("NiftiImageIO")
    if isMask:
        reader.SetOutputPixelType(sitk.sitkUInt8)
    elif isUInt:
        reader.SetOutputPixelType(sitk.sitkUInt16)
    reader.SetFileName(inputImageFileName)
    image = reader.Execute()
    return image

def arr_image_sitk(image):
    arr = sitk.GetArrayFromImage(image)
    return arr

def dsc_multivolume(p_dsc, output):
    logger.debug('Converting DSC to multivolume')
    imdsc, hdsc = nrrd.read(p_dsc)
    if 'space dimension' in hdsc.keys():
        _ = hdsc.pop('space dimension')
    if not 'space' in hdsc.keys():
        hdsc['space'] = 'left-posterior-superior'
    if not hdsc['kinds'][0]=='list':
        hdsc['kinds'][0] = 'list'
    if len(hdsc['space origin'])>3:
        imdsc = imdsc.transpose(3,0,1,2)
        hdsc['space directions'] = [None]+[hdsc['space directions'][i][:-1] for i in range(3)]
        hdsc['space origin'] = hdsc['space origin'][:-1]
    else:
        logger.debug('DSC volume seems already in multivolume format')
    nrrd.write(output,imdsc,hdsc)
    fs = ['MultiVolume.DICOM.EchoTime','MultiVolume.DICOM.FlipAngle','MultiVolume.FrameLabels','MultiVolume.FrameIdentifyingDICOMTagName','MultiVolume.FrameIdentifyingDICOMTagUnits','MultiVolume.NumberOfFrames']
    fcheck = [(f in hdsc.keys() and not hdsc[f]=='') for f in fs]
    dscfitvalid = all(fcheck)
    return dscfitvalid

def spacing_from_directions(sp_dir):
    spacing = list(np.round(np.sqrt(np.sum(sp_dir**2,axis=1)),2))
    return spacing

def print_im_figure(p_t1r, p_print, mapresult=None, cmapname=None, colorb=False, labels=None):
    import matplotlib.pyplot as plt
    imt1,_ = nrrd.read(p_t1r)
    if type(mapresult) is str:
        mapresult,_ = nrrd.read(mapresult)
        if not np.sum(mapresult)>1:
            logger.error('Segmentation seems empty!')
            mapresult = None
    def colorbar(mappable):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        last_axes = plt.gca()
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax)
        plt.sca(last_axes)
        return cbar
    os.makedirs(p_print,exist_ok=True)
    logger.debug('Output figure of probability map')
    if cmapname:
        try:
            mycmap = plt.cm.get_cmap(cmapname)
        except ValueError:
            cmapname = 'autumn'
    if mapresult is not None:
        zslices = [s for s in range(mapresult.shape[-1]) if (mapresult[:,:,s]>0).any()]
    else:
        zslices = [0, imt1.shape[-1]-1]
    plt.ioff()
    for i,largest in enumerate(range(min(zslices),max(zslices)+1,1)):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(np.rot90(imt1[:,:,largest],k=1), cmap=plt.cm.gray)
        if mapresult is not None:
            if colorb:
                alphas = 0.99
            else:
                alphas = np.clip(np.rot90(mapresult[:,:,largest],k=1), 0.01, 0.9)
            im = ax.imshow(np.rot90(mapresult[:,:,largest],k=1), cmap=mycmap, vmin=0, vmax=1, alpha=alphas)
            if colorb:
                cb = colorbar(im)
                cb.ax.tick_params(labelsize=16)
                if labels:
                    cb.ax.text(0, -0.05, labels[0], size=16)
                    cb.ax.text(0, 1.05, labels[-1], size=16)
        ax.axis('off')
        fig.tight_layout()
        num = '0'*(3-len(str(i+1)))+str(i+1)
        fig.savefig(os.path.join(p_print,num+'.png'), dpi = 300)
        plt.close(fig)

def write_map(probs, classes, p_lesion, p_output, printout=True, p_t1r=[], p_print=None, cmapname=None, labels=None):
    classlb = classes+1
    mask,h = nrrd.read(p_lesion)
    lbresult = np.zeros(mask.shape)
    r,c,z = np.where(mask==1)
    for i,_ in enumerate(r):
        lbresult[r[i],c[i],z[i]] = classlb[i]
    nrrd.write(os.path.join(p_output,'class-label.nrrd'),lbresult,h)
    mapresult = np.zeros(mask.shape)
    r,c,z = np.where(mask==1)
    for i,_ in enumerate(r):
        mapresult[r[i],c[i],z[i]] = probs[i,1]
    mapresult[mask==0] = np.nan
    nrrd.write(os.path.join(p_output,'probmap.nrrd'),mapresult,h)
    if printout:
        print_im_figure(p_t1r,p_print,mapresult,cmapname, colorb=True, labels=labels)

def mypdf(data):
    pdf = np.zeros(len(data))
    for h,val in enumerate(data):
        if h>0:
            pdf[h] = pdf[h-1]+val
        else:
            pdf[h] = val
    return pdf

def hist_perc(d, nbin=40):
    count, bins = np.histogram(d, nbin, density=False)
    count = count/np.sum(count)
    hperc = mypdf(count)
    return count, bins, hperc

def write_hist(probs, p_output, perc=False):
    import matplotlib.pyplot as plt
    plt.ioff()
    a=0.7
    b=np.arange(0, 1.1, 0.1)
    fig, ax = plt.subplots(figsize=(6, 4))
    if perc:
        count, bins, _ = hist_perc(probs,b)
        plt.hist(bins[:-1], bins, weights=count)
        ax.set_ylabel("% Voxels", size=20)
    else:
        ax.hist(probs, bins=b, alpha=a)
        ax.set_ylabel("n Voxels", size=20)
    ax.set_xlabel("Probability", size=20)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    fig.tight_layout()
    fig.savefig(p_output, dpi = 300)
    plt.close(fig)
