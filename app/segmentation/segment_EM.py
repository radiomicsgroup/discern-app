# -*- coding: utf-8 -*-
"""
Expectation-maximization brain segmentation
NOTE: if no graphical interface, needs xvfb-run
"""

import os
import time
from app.utils import get_exitcode_stdout_stderr, end_execution, wrap_string
import logging
logger = logging.getLogger(__name__)


def EMsegment(im_t1_path, output_path, slicer_exe, emsegcli, mrml):
    if not output_path:
        output_path = os.path.dirname(im_t1_path)
    outfolder = os.path.join(output_path)
    if not os.path.isdir(outfolder): os.makedirs(outfolder, exist_ok=True)
    
    logger.info('Segmenting hemispheres')
    cli_exe = 'xvfb-run -a ' + slicer_exe + ' --launch ' + emsegcli
    param_io = ['--mrmlSceneFileName','--resultVolumeFileName','--targetVolumeFileNames']
    
    files_io = [mrml, os.path.join(outfolder,'HEMS-label.nrrd'), im_t1_path]
    files_io = [wrap_string(f,'"') for f in files_io]
    all_io = ' '.join([' '.join([p, files_io[i]]) for i,p in enumerate(param_io)])
    params_str = '--taskPreProcessingSetting :C1:C0 --intermediateResultsDirectory '+wrap_string(outfolder,'"')+' --disableMultithreading -1 --dontUpdateIntermediateData -1 --registrationAffineType 1 --registrationDeformableType 0'
    call_str = " ".join([cli_exe, all_io, params_str])
    logger.debug('CMD call: %s', call_str)
    try:
        exitcode, out, err = get_exitcode_stdout_stderr(call_str)
    except FileNotFoundError as e:
        logger.warning('xvfb-run was not detected. If on Linux, please install it!')
        if 'xvfb-run' in e.filename:
            logger.debug('Attempting to run without xvfb-run...')
            cli_exe = slicer_exe + ' --launch ' + emsegcli
            call_str = " ".join([cli_exe, all_io, params_str])
            logger.debug('CMD call: %s', call_str)
            exitcode, out, err = get_exitcode_stdout_stderr(call_str)
            logger.debug('CMD output:\n%s',out.decode())
            if exitcode:
                logger.error('Terminated with error code:\n%s',err.decode())
                end_execution()
        else:
            raise e
    logger.debug('CMD output:\n%s',out.decode())
    if exitcode:
        logger.error('Terminated with error code.\n%s',err.decode())
    timeout = time.time() + 30
    while((not os.path.exists(os.path.join(outfolder,'HEMS-label.nrrd'))) and time.time()<timeout):
        logger.debug('Waiting for HEMS file writer to finish...')
        time.sleep(4)
    if not os.path.exists(os.path.join(outfolder,'HEMS-label.nrrd')):
        logger.error('HEMS file writer timeout! Slicer HEMS failed')
        end_execution()
    return os.path.join(outfolder,'HEMS-label.nrrd')
