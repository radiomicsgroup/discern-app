# -*- coding: utf-8 -*-
"""
preprocess DSC
"""

import os
import numpy as np
import nrrd
from app.utils import get_exitcode_stdout_stderr, end_execution, wrap_string

import logging
logger = logging.getLogger(__name__)


def sk_strip_perf(imp,noStrip):
    from skimage import morphology, filters
    th_forep = filters.threshold_li(imp)
    maskp = imp>th_forep
    if noStrip:
        imp_masked = imp*maskp
        return imp_masked, maskp
    else:
        se2 = np.zeros((9,9,9))
        se2[:,:,4] = 1
        se = morphology.cube(5)
        mask_per = morphology.binary_erosion(maskp, se)
        p_size = sum(sum(sum(mask_per)))
        mask_pcl = morphology.remove_small_objects(mask_per,p_size/2,1)
        mask_pdi = morphology.binary_dilation(mask_pcl, se)
        imp_masked = imp*mask_pdi
        return imp_masked, mask_pdi

def get_ttp_perfusion(im_perf):
    m = []
    for t in range(im_perf.shape[0]):
        m.append(np.mean(im_perf[t,:,:,:]))
    ttp = np.argmin(m)
    return ttp

def get_dsc_snaps(im_perf,h_perf):
    if len(im_perf.shape)<4:
        logger.error('Input DSC image is not 4D!')
        end_execution()
    ttp = get_ttp_perfusion(im_perf)
    h_frame = {
        'space': 'left-posterior-superior',
        'kinds': ['domain', 'domain', 'domain'],
        'space directions': h_perf['space directions'][1:,:],
        'space origin': h_perf['space origin']
        }
    return im_perf[0,:,:,:], im_perf[ttp,:,:,:], h_frame

def read_write_dsc_snaps(p_dsc,output_path,noStrip):
    im_perf, h_perf = nrrd.read(p_dsc)
    t0, tref, h = get_dsc_snaps(im_perf,h_perf)
    nrrd.write(os.path.join(output_path,'DSC_t0.nrrd'),t0,h)
    nrrd.write(os.path.join(output_path,'DSC_ref.nrrd'),tref,h)
    imp_masked, mask_brain = sk_strip_perf(t0,noStrip)
    nrrd.write(os.path.join(output_path, 'BRAIN-label.nrrd'), np.array(mask_brain,dtype='uint8'), h)
    nrrd.write(os.path.join(output_path, 'DSC_sk.nrrd'), imp_masked, h)
    return os.path.join(output_path,'DSC_t0.nrrd'), os.path.join(output_path,'DSC_ref.nrrd'), os.path.join(output_path, 'DSC_sk.nrrd'), os.path.join(output_path,'BRAIN-label.nrrd')

def correct_DSC(p_dscref, input_path, output_path, slicer_exe):
    logger.info('correcting DSC')
    if not output_path:
        output_path = os.path.dirname(input_path)
    temppath = os.path.join(output_path,'MoCo.tmp')
    os.makedirs(temppath, exist_ok=True)
    im_perf, h_perf = nrrd.read(input_path)
    ttp = get_ttp_perfusion(im_perf)
    
    time = [int(float(v)) for v in h_perf['MultiVolume.FrameLabels'].split(',')]
    dupid = np.array([i for i,n in enumerate(time) if n in time[:i]])
    
    news = np.array(im_perf.shape)
    news[3] = news[3]-len(dupid)
    new_img = np.zeros(news, dtype='uint16')
    h_frame = {
        'space': 'left-posterior-superior',
        'kinds': ['domain', 'domain', 'domain'],
        'space directions': h_perf['space directions'][1:,:],
        'space origin': h_perf['space origin'][:]
        }
    logger.debug('registering DSC frames')
    new_img[ttp,:,:,:] = im_perf[ttp,:,:,:]
    for t in range(im_perf.shape[0]):
        if not t==ttp and not t in dupid:
            nrrd.write(os.path.join(temppath,'temp.nrrd'),im_perf[t,:,:,:],h_frame)
            savename = os.path.join(temppath,'reg')
            cli_exe = slicer_exe + ' --launch BRAINSFit'
            param_io = ['--fixedVolume','--movingVolume','--linearTransform','--outputVolume']
            files_io = [p_dscref, os.path.join(temppath,'temp.nrrd'), savename+'.tfm', savename+'.nrrd']
            files_io = [f.replace('\\','/') for f in files_io]
            files_io = [wrap_string(f,'"') for f in files_io]
            all_io = ' '.join([' '.join([p, files_io[i]]) for i,p in enumerate(param_io)])
            params_str = '--samplingPercentage 0.1 --splineGridSize 5,5,5 --initializeTransformMode Off --useAffine --maskProcessingMode NOMASK --medianFilterSize 0,0,0 --removeIntensityOutliers 0 --outputVolumePixelType float --backgroundFillValue 0 --interpolationMode BSpline --numberOfIterations 1000 --maximumStepLength 0.05 --minimumStepLength 0.001 --relaxationFactor 0.5 --translationScale 50 --reproportionScale 1 --skewScale 1 --maxBSplineDisplacement 0 --fixedVolumeTimeIndex 0 --movingVolumeTimeIndex 0 --numberOfHistogramBins 20 --numberOfMatchPoints 10 --costMetric MMI --maskInferiorCutOffFromCenter 1000 --ROIAutoDilateSize 0 --ROIAutoClosingSize 9 --numberOfSamples 0 --failureExitCode -1 --numberOfThreads -1 --debugLevel 0 --costFunctionConvergenceFactor 2e+13 --projectedGradientTolerance 1e-05 --maximumNumberOfEvaluations 900 --maximumNumberOfCorrections 25 --metricSamplingStrategy Random'
            call_str = " ".join([cli_exe, all_io, params_str])
            logger.debug('CMD call: %s', call_str)
            exitcode, out, err = get_exitcode_stdout_stderr(call_str)
            if exitcode:
                logger.error('Registration terminated with error code: %s',err.decode())
                end_execution()
            else:
                logger.debug('Registration ok: %s', out.decode())
                frame,_ = nrrd.read(savename+'.nrrd')
                frame[frame<0] = 0
                frame = frame.astype('uint16')
                new_img[t,:,:,:] = frame
    logger.debug('saving fixed DSC')
    nrrd.write(os.path.join(output_path,'DSC_fix.nrrd'), new_img, h_perf)
    return os.path.join(output_path,'DSC_fix.nrrd')
 
