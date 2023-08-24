# -*- coding: utf-8 -*-
"""
wrapper for resampling and registration with Slicer
"""

import os
import numpy as np
from ..inout.inout import spacing_from_directions
from app.utils import get_exitcode_stdout_stderr, cast_to_type, end_execution, wrap_string
import nrrd

import logging
logger = logging.getLogger(__name__)

def resamp(movim, refim, outim, tf, isLabel, slicer_exe, spacing=[]):
    logger.debug('Resampling')
    if spacing:
        logger.debug('Only spacing resampling')
        if not spacing==True:
            starget = spacing
        else:
            _,h = nrrd.read(refim)
            starget = spacing_from_directions(h['space directions'])
        starget = ','.join([str(s) for s in starget])
        cli_exe = slicer_exe + ' --launch ResampleScalarVolume'
        files_io = [movim, outim]
        files_io = [f.replace('\\','/') for f in files_io]
        files_io = [wrap_string(f,'"') for f in files_io]
        all_io = ' '.join(files_io)
        if isLabel:
            params_str = '--spacing '+starget+' --interpolation nearestNeighbor'
        else:
            params_str = '--spacing '+starget+' --interpolation linear'
        call_str = " ".join([cli_exe, all_io, params_str])
        logger.debug('CMD call: %s', call_str)
        exitcode, out, err = get_exitcode_stdout_stderr(call_str)
        logger.debug('%s', out.decode())
        if exitcode==1:
            logger.error('Terminated with error code:\n%s', err.decode())
            end_execution()
    else:
        cli_exe = slicer_exe + ' --launch BRAINSResample'
        if tf:
            param_io = ['--inputVolume','--referenceVolume','--outputVolume', '--warpTransform']
            files_io = [movim, refim, outim, tf]
        else:
            param_io = ['--inputVolume','--referenceVolume','--outputVolume']
            files_io = [movim, refim, outim]
        files_io = [f.replace('\\','/') for f in files_io]
        files_io = [wrap_string(f,'"') for f in files_io]
        all_io = ' '.join([' '.join([p, files_io[i]]) for i,p in enumerate(param_io)])
        if isLabel:
            params_str = '--pixelType uint --interpolationMode NearestNeighbor --defaultValue 0 --numberOfThreads -1'
        else:
            params_str = '--pixelType float --interpolationMode Linear --defaultValue 0 --numberOfThreads -1'
        call_str = " ".join([cli_exe, all_io, params_str])
        logger.debug('CMD call: %s', call_str)
        exitcode, out, err = get_exitcode_stdout_stderr(call_str)
        logger.debug('%s', out.decode())
        if exitcode==1:
            logger.error('Terminated with error code:\n%s', err.decode())
            end_execution()

def reg_T1_to_DSC(im_t1_path, im_dscref_path, output_path, slicer_exe):
    if not output_path:
        output_path = os.path.join(os.path.dirname(im_t1_path),'reg')
    outfolder = output_path
    if not os.path.isdir(outfolder): os.makedirs(outfolder, exist_ok=True)
    logger.info('Registering T1C')
    p_out = os.path.join(outfolder,'T1C_reg.nrrd')
    p_tfm = p_out.replace('.nrrd','.tfm')
    cli_exe = slicer_exe + ' --launch BRAINSFit'
    param_io = ['--fixedVolume','--movingVolume','--linearTransform', '--outputVolume']
    files_io = [os.path.join(im_dscref_path), os.path.join(im_t1_path), p_tfm, p_out]
    files_io = [f.replace('\\','/') for f in files_io]
    files_io = [wrap_string(f,'"') for f in files_io]
    all_io = ' '.join([' '.join([p, files_io[i]]) for i,p in enumerate(param_io)])
    params_str = '--initializeTransformMode Off --samplingPercentage 0.95 --useRigid --useAffine --splineGridSize 10,10,10 --numberOfIterations 2000 --maskProcessingMode NOMASK --outputVolumePixelType float --backgroundFillValue 0 --maskInferiorCutOffFromCenter 1000 --interpolationMode BSpline --minimumStepLength 0.001 --translationScale 1000 --reproportionScale 0 --skewScale 0 --maxBSplineDisplacement 0 --numberOfHistogramBins 50 --numberOfMatchPoints 10 --numberOfSamples 0 --fixedVolumeTimeIndex 0 --movingVolumeTimeIndex 0 --medianFilterSize 0,0,0 --removeIntensityOutliers 0 --ROIAutoDilateSize 0 --ROIAutoClosingSize 9 --relaxationFactor 0.5 --maximumStepLength 0.05 --failureExitCode -1 --numberOfThreads -1 --debugLevel 0 --costFunctionConvergenceFactor 2e+013 --projectedGradientTolerance 1e-005 --maximumNumberOfEvaluations 900 --maximumNumberOfCorrections 25 --metricSamplingStrategy Random --costMetric MMI'
    call_str = " ".join([cli_exe, all_io, params_str])
    logger.debug('CMD call: %s', call_str)
    exitcode, out, err = get_exitcode_stdout_stderr(call_str)
    logger.debug('CMD output:\n%s',out.decode())
    if exitcode==1:
        logger.error('Terminated with error code:\n%s', err.decode())
        end_execution()
    
    im2,h2 = nrrd.read(p_out)
    _,h1 = nrrd.read(im_t1_path)
    im2 = cast_to_type(im2, h1['type'])
    h2['type'] = h1['type']
    nrrd.write(p_out,im2,h2)
    return p_out,p_tfm
