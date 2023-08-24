# -*- coding: utf-8 -*-
"""
DICOM conversion to Slicer-compatible NRRD, using dcm2niix
"""

import logging
logger = logging.getLogger(__name__)

import os
import shlex
from subprocess import Popen, PIPE
from .inout import split_filename, read_image_sitk, write_image_sitk
from app.utils import end_execution, wrap_string

def num28bit(a):
    b = bin(a).replace('0b','')
    x = b[::-1]
    while len(x) < 8:
        x += '0'
    b = x[::-1]
    return b

def get_exitcode_stdout_stderr(cmd):
    """
    Execute the external command and get its exitcode, stdout and stderr.
    """
    args = shlex.split(cmd)
    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    return exitcode, out, err

def dcm_decomp(inpath, outpath):
    from pydicom import dcmread
    for root, dirs, files in os.walk(inpath):
        for file in files:
            ds = dcmread(os.path.join(root, file),force=True)
            ds.decompress()
            ds.save_as(os.path.join(outpath,file))

def DICOM2nii(inputpath, outputpath, outfilename=[], dcm2niix_exe=[]):
    '''
    dcm2niix is called here. dcm2niix binary must be downloaded and the path given in dcm2niix_exe argument.
    
    Use dcm2niix following its licensing below (https://github.com/rordenlab/dcm2niix/):
    
    LICENSING
    Copying and distribution of this file, with or without modification, are permitted in any medium without royalty provided the copyright notice and this notice are preserved. This file is offered as-is, without any warranty. The dcm2niix project is distributed under the BSD 2-Clause License.
    COPYRIGHT
    2022 The dcm2niix contributors
    
    '''
    logger.info('dcm2niix dicom conversion')
    if not outfilename:
        outfilename = os.path.basename(inputpath)
    if not os.path.exists(outputpath): os.makedirs(outputpath)
    dir_io = ' '.join([wrap_string(outputpath,'"'), wrap_string(inputpath,'"')])
    params_str = ' '.join(['-p', 'n', '-e', 'n', '-w', '1', '--terse', '-f', wrap_string(outfilename,'"'), '-o'])
    call_str = " ".join([dcm2niix_exe, params_str, dir_io])
    logger.debug('Calling dcm2niix: %s', call_str)
    exitcode, out, err = get_exitcode_stdout_stderr(call_str)
    logger.debug('%s',out.decode())
    if exitcode:
        logger.error('dcm2niix error:\n%s',err.decode())
        if 'Error: Compressed' in err.decode():
            logger.info('DICOM seems compressed. Attempting to decompress...')
            pdecomp = os.path.join(os.path.dirname(outputpath), os.path.basename(inputpath)+'.decomp')
            os.makedirs(pdecomp)
            dcm_decomp(inputpath, pdecomp)
            dir_io = ' '.join([wrap_string(outputpath,'"'), wrap_string(pdecomp,'"')])
            call_str = " ".join([dcm2niix_exe, params_str, dir_io])
            logger.debug('Calling dcm2niix: %s', call_str)
            exitcode, out, err = get_exitcode_stdout_stderr(call_str)
            logger.debug('%s',out.decode())
            if exitcode:
                logger.error('dcm2niix error:\n%s',err.decode())
                end_execution()
        else:
            end_execution()
    return os.path.join(outputpath,outfilename+'.nii')

def json2dict(p_json):
    """converts single json to dictionary"""
    import json
    with open(p_json, 'r') as f:
        s = f.read()
    s = s.replace(' inf,',' Infinity,')
    s = s.replace(' nan,',' NaN,')
    dct = json.loads(s)
    return dct

def convert2nrrd(inputpath, outputpath, isUInt, jsoninfo=None):
    logger.info('sitk nii to nrrd conversion')
    im = read_image_sitk(inputpath, isUInt)
    if any(['Multivolume' for k in im.GetMetaDataKeys()]):
        logger.info('Slicer multivolume detected - skipping compatibility changes')
        return inputpath
    if im.GetDimension()==4:
        if jsoninfo:
            jdict = json2dict(jsoninfo)
            if 'RepetitionTime' in jdict:
                im.SetMetaData('MultiVolume.DICOM.RepetitionTime', str(jdict['RepetitionTime']*1000))
                t0 = jdict['RepetitionTime']
            else:
                t0 = float(im.GetMetaData('pixdim[4]'))
                logger.warning('TR not found within imaging parameters! Continuing with value from temporal dimension: %s', t0)
            if 'RepetitionTimeExcitation' in jdict:
                t01 = jdict['RepetitionTimeExcitation']
                if not t0==t01:
                    if float(t0)>float(t01):
                        logger.warning('Detected TR > RepetitionTimeExcitation (multi volume acq. timing). Using TR: %s', t0)
                    else:
                        logger.warning('Detected TR < RepetitionTimeExcitation (multi volume acq. timing). Using RepetitionTimeExcitation: %s', t01)
                        t0=t01
            if 'EchoTime' in jdict:
                im.SetMetaData('MultiVolume.DICOM.EchoTime', str(jdict['EchoTime']*1000))
            else:
                logger.warning('TE not found! Continuing, but DSC fit will fail')
            if 'FlipAngle' in jdict:
                im.SetMetaData('MultiVolume.DICOM.FlipAngle', str(jdict['FlipAngle']))
            else:
                logger.warning('Flip angle not found! Continuing, but DSC fit will fail')
        else:
            logger.warning('Imaging parameters not found! Continuing, but DSC fit will fail')
            t0 = float(im.GetMetaData('pixdim[4]'))
            logger.warning('TR not found! Continuing with value from temporal dimension: %s', t0)
        tlen = int(im.GetMetaData('dim[4]'))
        u0 = im.GetMetaData('xyzt_units')
        u1 = num28bit(int(u0))
        u1_t = u1[-6:-3]
        if u1_t=='001':
            logger.debug('time units are seconds, convert to milliseconds')
            t0 = t0*1000
        elif u1_t=='010':
            logger.debug('time units are milliseconds')
        else:
            logger.warning('unknown time units!')
            if t0<10:
                logger.debug('assuming time units are seconds, convert to milliseconds')
                t0 = t0*1000
            else:
                logger.debug('assuming time units are milliseconds')
        t = [str(int(round(t0*i))) for i in range(tlen)]
        im.SetMetaData('MultiVolume.FrameLabels', ','.join(t))
        im.SetMetaData('MultiVolume.FrameIdentifyingDICOMTagName', 'AcquisitionTime')
        im.SetMetaData('MultiVolume.FrameIdentifyingDICOMTagUnits', 'ms')
        im.SetMetaData('MultiVolume.NumberOfFrames', str(len(t)))
    write_image_sitk(outputpath, im, isUInt)
    return outputpath


def DICOM2nrrd(inputpath, outputpath, outfilename, dcm2niix_exe):
    from shutil import copyfile
    p, b, e = split_filename(inputpath)
    if 'nii' in e or 'nrrd' in e:
        logger.warning('Detected non-DICOM input file. Skipping conversion')
        newfilename = os.path.join(outputpath,outfilename+'.nrrd')
        path, base, ext = split_filename(inputpath)
        if os.path.exists(inputpath.replace(ext,'.json')):
            outfilename = convert2nrrd(inputpath, newfilename, isUInt=True, jsoninfo=inputpath.replace(ext,'.json'))
        else:
            outfilename = convert2nrrd(inputpath, newfilename, isUInt=True)
        return outfilename,None
    else:
        outfile = DICOM2nii(inputpath, outputpath, outfilename, dcm2niix_exe)
        path, base, ext = split_filename(outfile)
        newfilename = os.path.join(path, base+'.nrrd')
        outfilename = convert2nrrd(outfile, newfilename, isUInt=True, jsoninfo=outfile.replace(ext,'.json'))
        return outfilename,outfile
