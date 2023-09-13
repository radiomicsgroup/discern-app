# -*- coding: utf-8 -*-
"""

This is the code for the software DISCERN.
This software is property of The Vall d'Hebron Institute of Oncology (VHIO), Barcelona, Spain.

Terms of Use: 
1.	The use of any software provided by VHIO as part of its services, from now on referred to as the services, is subject to the terms and conditions of the third party license agreements applicable to the relevant software as made available through the website [https://github.com/radiomicsvhio/discern-app].
2.	The services shall be used solely for research activities that comply with all applicable laws and regulations.
3.	Users shall keep all user IDs, passwords and other means of access to the services account within its possession or control and shall keep those confidential and secure from unauthorised access or use.
4.	If user becomes aware of any unauthorized use of a password or the services account, user will notify the VHIO team as promptly as possible through the provided channels.
5.	User will not use the services to identify the individuals who are data subjects.
6.	Do NOT use any of the services if the safety of data subjects depends directly on the uninterrupted, and error free availability of the services at all times, or the compatibility or operation of the services with all hardware and software configurations.
7.	Within the context of a study, the principal investigator is responsible for:
    - Verification of the identity of users and delegates;
    - Order assignment and withdrawal of access authorizations for users and delegates;
    - Informing users and delegates about the responsible handling of study data.
8.	Within the context of a study, users may only use the services as per instructions of the study’s principal investigator, specifically:
    - Data upload and download to and from the services shall only be executed in accordance with the study conditions and with the consent of the principal investigator.

For further information and contact, please visit:
https://radiomicsgroup.github.io/
https://github.com/radiomicsvhio

"""

from app.segmentation import segment_EM, segment_T1, segment_WM
from app.processing import preprocess, preprocess_DSC, reg_imgs, dsc_curves
from app.inout.inout import dsc_multivolume, print_im_figure
from app.inout import dicom_convert
from app import utils, class_launcher
import settings
import tensorflow as tf
import argparse
import time
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

APPVERSION = '1.0.7'


def main(p_dsc, p_t1, p_output, p_model1, p_model2, threshold_lym, threshold_met, p_lesion0, p_wm0, rerunMask, noReg, noMoCo, noStrip, logfile=None):
    logger = logging.getLogger(__name__)
    resamp_spacing = [1,1,3]
    p_result = os.path.join(p_output, 'results')
    if not os.path.exists(p_result):
        os.makedirs(p_result)
    p_im = os.path.join(p_output, 'im.tmp')
    if not os.path.exists(p_im):
        os.makedirs(p_im)
    regoutput = os.path.join(p_output, 'reg')

    run_class = True
    if not rerunMask:
        logger.debug('Creating output folder')
        os.makedirs(regoutput, exist_ok=True)
        
        p_dsc2, _ = dicom_convert.DICOM2nrrd(p_dsc, p_im, outfilename='DSC', dcm2niix_exe=settings.DCM2NIIX)
        p_dscmv = os.path.join(p_im, 'DSC_slicer.nrrd')
        _ = dsc_multivolume(p_dsc2, p_dscmv)
        _, p_dscref, _, p_brain = preprocess_DSC.read_write_dsc_snaps(p_dscmv, p_im, noStrip)
        if not noMoCo:
            p_dscmv = preprocess_DSC.correct_DSC(p_dscref, p_dscmv, output_path=p_im, slicer_exe=settings.SLICER_EXEC)
        else:
            p_dscmv = p_dscmv
        
        if p_t1 and not (p_lesion0 and p_wm0):
            p_t12, _ = dicom_convert.DICOM2nrrd(p_t1, p_im, outfilename='T1C', dcm2niix_exe=settings.DCM2NIIX)
            p_t13, _ = preprocess.correct_T1(p_t12, output_path=p_im, slicer_exe=settings.SLICER_EXEC)
            p_t1r = os.path.join(regoutput, 'T1C_reg.nrrd')
            if noReg:
                if not os.path.exists(p_t1r): p_t1r = p_t13
            else:
                _, p_tfm = reg_imgs.reg_T1_to_DSC(p_t13, p_dscref, output_path=regoutput, slicer_exe=settings.SLICER_EXEC)
            reg_imgs.resamp(p_t13, p_dscref, p_t1r, None, False, settings.SLICER_EXEC, spacing=resamp_spacing)
            reg_imgs.resamp(p_brain, p_t1r, p_brain, None, True, settings.SLICER_EXEC)
            p_hems = segment_EM.EMsegment(p_t1r, output_path=regoutput, slicer_exe=settings.SLICER_EXEC, emsegcli=settings.EMSEGMENTCOMMANDLINE, mrml=settings.MRML)
            p_brain = segment_T1.and_brain_hems(p_hems, p_brain)
        elif not p_t1:
            logger.warning('Forcing noReg as T1 not provided')
            noReg = True
        else:
            logger.warning('Disregarding T1 as all masks provided')
            p_t1 = None
            noReg = True
        
        if p_lesion0:
            p_lesion = dicom_convert.convert2nrrd(p_lesion0, os.path.join(regoutput, 'lesion-label.nrrd'), isUInt=True)
            if p_t1:
                reg_imgs.resamp(p_lesion, p_t1r, p_lesion, None, True, settings.SLICER_EXEC)
            else:
                reg_imgs.resamp(p_lesion, p_dscref, p_lesion, None, True, settings.SLICER_EXEC, spacing=resamp_spacing)
        else:
            if p_t1:
                p_lesion, _ = segment_T1.segment(p_t1r, p_brain, p_hems, output_path=[])
            else:
                run_class = False
                p_lesion = p_brain
        
        if p_wm0:
            p_wm = dicom_convert.convert2nrrd(p_wm0, os.path.join(regoutput, 'WM-label.nrrd'), isUInt=True)
            if p_t1:
                reg_imgs.resamp(p_wm, p_t1r, p_wm, None, True, settings.SLICER_EXEC)
            else:
                reg_imgs.resamp(p_wm, p_dscref, p_wm, None, True, settings.SLICER_EXEC, spacing=resamp_spacing)
        else:
            p_wm = segment_WM.segment(p_t1r, p_brain, p_hems, p_lesion, output_path=[])
    else:
        p_dscmv = os.path.join(p_im, 'DSC_slicer.nrrd')
        p_dscref = os.path.join(p_im, 'DSC_ref.nrrd')
        p_t1r = os.path.join(regoutput, 'T1C_reg.nrrd')
        p_brain = os.path.join(regoutput, 'BRAIN-label.nrrd')
        p_hems = os.path.join(regoutput, 'HEMS-label.nrrd')
        if not noReg:
            p_tfm = os.path.join(regoutput, 'T1C_reg.tfm')
        if p_lesion0:
            p_lesion = dicom_convert.convert2nrrd(p_lesion0, os.path.join(regoutput, 'lesion-label.nrrd'), isUInt=True)
        else:
            p_lesion, _ = segment_T1.segment(p_t1r, p_brain, p_hems, output_path=[])
        if p_wm0:
            p_wm = dicom_convert.convert2nrrd(p_wm0, os.path.join(regoutput, 'WM-label.nrrd'), isUInt=True)
        else:
            p_wm = segment_WM.segment(p_t1r, p_brain, p_hems, p_lesion, output_path=[])
    if noReg:
        reg_imgs.resamp(p_lesion, p_dscref, p_lesion, None, True, settings.SLICER_EXEC)
        reg_imgs.resamp(p_wm, p_dscref, p_wm, None, True, settings.SLICER_EXEC)
    else:
        reg_imgs.resamp(p_lesion, p_dscref, p_lesion, p_tfm, True, settings.SLICER_EXEC)
        reg_imgs.resamp(p_wm, p_dscref, p_wm, p_tfm, True, settings.SLICER_EXEC)
    if p_t1:
        p_printim = p_t1r
        if noReg:
            reg_imgs.resamp(p_t1r, p_dscref, p_t1r, None, False, settings.SLICER_EXEC)
        else:
            reg_imgs.resamp(p_t1r, p_dscref, p_t1r, p_tfm, False, settings.SLICER_EXEC)
    else:
        p_printim = p_dscref
    try:
        print_im_figure(p_printim, p_print=os.path.join(p_result, 'ref_im'))
        print_im_figure(p_printim, p_print=os.path.join(p_result, 'seg_lesion'), mapresult=p_lesion, cmapname='Wistia')
        print_im_figure(p_printim, p_print=os.path.join(p_result, 'seg_wm'), mapresult=p_wm, cmapname='cool_r')
    except Exception as e:
        logger.error('Could not print figures! %s',e)

    p_curves, _ = dsc_curves.extract_norm_curves(p_dscmv, p_lesion, p_wm, output_path=p_output, out_plot=True, in_slicer_shape=True, p_print=p_result)
    
    if not os.path.exists(p_result):
        os.makedirs(p_result)

    if run_class:
        res = class_launcher.data_class_task(p_curves, p_output, p_model1, p_model2, p_lesion,threshold_lym, threshold_met, p_printim, printout=True, p_print=p_result)
        logger.debug('%s', res)
    
    logger.debug('Creating JSON file')
    utils.create_json(p_result, 'display.json', res, [logfile])

def get_file_from_folder(p_input):
    if p_input and os.path.exists(p_input):
        if os.path.isdir(p_input):
            flist = utils.listfiles(p_input)
            if len(flist)==1:
                p_input = os.path.join(p_input,flist[0])
            elif len(flist)==0:
                logger.debug('No files in dir')
                p_input = None
            else:
                raise Exception('More than 1 file found under %s', p_input)
    else:
        p_input = None
    return p_input





if __name__ == "__main__":

    timestr = time.strftime("%Y%m%d-%H%M%S")
    logpath = settings.LOG_FOLDER
    os.makedirs(logpath, exist_ok=True)
    logfile = os.path.join(logpath, timestr+'.log')

    logging.basicConfig(
        filename=logfile,
        filemode='a',
        level=logging.DEBUG,
        format='-- %(asctime)s %(name)-30s %(funcName)-20s line %(lineno)-3s -- %(levelname)-8s : %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    if settings.DEBUG:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter('%(name)-30s -- %(levelname)-8s : %(message)s'))
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)
    logger.addHandler(console)
    logger.propagate = 0
    utils.Tee(logfile, 'a')
    logger.info('-'*20 + 'Starting event logging' + '-'*20)

    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.colorbar').disabled = True
    tf.get_logger().setLevel('ERROR')

    logger.debug('Parsing arguments')
    parser = argparse.ArgumentParser(prog='DSC processing and tumor classification',
                                     description='''\
            Process DSC brain images of GBM, PCNSL or metastases and applies pre-trained CNN tumour classifier.
            ''')
    parser.add_argument('--p_dsc', action='store', type=os.path.normpath, help='DSC image file path. DICOM, Nifti and NRRD formats accepted.')
    parser.add_argument('--p_t1', action='store', type=os.path.normpath, help='T1wCE image file path - can be skipped if lesion and WM masks are provided. DICOM, Nifti and NRRD formats accepted.')
    parser.add_argument('--p_output', action='store', default=os.path.join('.',timestr), type=os.path.normpath, help='Output path. Default: current date and time folder')
    parser.add_argument('--th_lym', action='store', default=60, type=float, help='Optional. Threshold for PCNSL vs non-PCNSL, range [0 100]. Please refer to the publication for more details')
    parser.add_argument('--th_met', action='store', default=84, type=float, help='Optional. Threshold for metastasis vs GBM, range [0 100]. Please refer to the publication for more details')
    parser.add_argument('--p_lesion', action='store', help='Optional. Enhancing lesion segmentation image file path. Image will undergo same resampling as T1 unless --noReg is specified. Nifti and NRRD formats accepted.')
    parser.add_argument('--p_wm', action='store', help='Optional. White matter (WM) segmentation image file path. Image will undergo same resampling as T1 unless --noReg is specified. Nifti and NRRD formats accepted.')
    parser.add_argument('--rerunMask', default=False, action='store_true', help='Optional. Warning: this option is still in development! Flag to rerun classification with custom mask(s). It uses previously preprocessed files')
    parser.add_argument('--noReg', default=False, action='store_true', help='Optional. Flag to skip registration of T1 (and masks) to DSC. Volumes will still be resampled to DSC space')
    parser.add_argument('--noMoCo', default=False, action='store_true', help='Optional. Flag to skip motion correction of DSC. Use if there is no movement in DSC to save processing time')
    parser.add_argument('--noStrip', default=False, action='store_true', help='Optional. Flag to skip skull stripping.')

    logger.info('Start Neuro Version ' + APPVERSION)

    args = parser.parse_args()
    try:
        p_dsc = os.path.normpath(args.p_dsc.strip('"'))
        assert os.path.exists(p_dsc)
        if args.p_t1:
            p_t1 = os.path.normpath(args.p_t1.strip('"'))
            if not os.path.exists(p_t1):
                p_t1 = None
        else:
            p_t1 = None
        p_output = args.p_output
        th_lym = args.th_lym
        th_met = args.th_met
        p_lesion = get_file_from_folder(args.p_lesion)
        p_wm = get_file_from_folder(args.p_wm)
        rerunMask = args.rerunMask
        noReg = args.noReg
        noMoCo = args.noMoCo
        noStrip = args.noStrip
    except TypeError as e:
        logger.error('Error in argument')
        raise e
    
    if not p_t1:
        logger.debug('No existing path for T1')
        p_t1 = None
        if not p_wm or not os.path.exists(p_wm) or not p_lesion or not os.path.exists(p_lesion):
            raise Exception('Not enough inputs. Need either T1 (auto segmentation), T1+ROI, T1+WM or ROI+WM (manual segmentation).')
    
    p_model1 = os.path.normpath(settings.MODEL1)
    p_model2 = os.path.normpath(settings.MODEL2)

    logger.debug('Calling main function')
    main(p_dsc, p_t1, p_output, p_model1, p_model2, th_lym, th_met, p_lesion, p_wm, rerunMask, noReg, noMoCo, noStrip, logfile)

    logger.handlers.clear()
    logging.shutdown()
    utils.end_execution()
