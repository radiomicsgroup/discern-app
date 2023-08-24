# -*- coding: utf-8 -*-
"""
DSC curve processing
"""

import logging
logger = logging.getLogger(__name__)

import os
import numpy as np
import nrrd
from scipy import signal
import matplotlib.pyplot as plt
from app.utils import end_execution


def listfolders(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not 'seg' in d]

def listfiles(folder):
    return [d for d in os.listdir(folder) if not os.path.isdir(os.path.join(folder, d))]

def check_mask_curve(min_point):
    if min_point>=-1:
        flag_fail = True
    else:
        flag_fail = False
    return flag_fail

def check_mean_perfusion(im_perf):
    m1 = list(np.mean(im_perf, axis=(0,1,2)))
    ttp = np.argmin(m1)
    if not (ttp>2 and ttp<(len(m1)-3)):
        logger.debug('TTP not found!')
        ttp = None
    return m1, ttp

def is_bolus_detected(y_filtb, g):
    y_norm = (y_filtb-np.amax(y_filtb))/-np.min(y_filtb-np.amax(y_filtb))
    cr = signal.correlate(y_norm,g,mode='same')
    if np.amax(cr)>=3:
        return (True, y_filtb)
    else:
        return (False, y_filtb)


def filter_curves(im_perf, im_mask, time, use_bolus_det=0, use_filt=1):
    if use_filt:
        sf = 1/time[1]
        low_freq_th = 1/10
        b, a = signal.butter(1, low_freq_th/(sf*0.5), btype = 'lowpass', output='ba')
    if use_bolus_det:
        g = -(signal.gaussian(40,2)-0.0)[10:-10]
    r,c,z = np.where(im_mask==1)
    curve_array = []
    new_mask = np.zeros(np.shape(im_mask))
    pixcount = 0
    for i,_ in enumerate(r):
        y = im_perf[r[i],c[i],z[i],:]
        if use_filt:
            y_b = signal.medfilt(y,3)
            y_filtb = signal.filtfilt(b, a, y_b)
        else:
            y_filtb = y
        if use_bolus_det:
            out_pixel = is_bolus_detected(y_filtb,g)
        else:
            out_pixel = [True, y_filtb]
        if out_pixel[0]:
            curve_array.append(out_pixel[1])
            pixcount += 1
            new_mask[r[i],c[i],z[i]] = 1
    return [curve_array, new_mask]


def get_curve_peaks(curve_avg):
    ttp = np.argmin(curve_avg)
    if ttp==0 or ttp>(len(curve_avg)-3):
        curve_avg[0] = np.mean(curve_avg[:3])
        curve_avg[-3:] = np.mean(curve_avg[-5:])
        ttp = np.argmin(curve_avg)
    if ttp>2 and ttp<(len(curve_avg)-3):
        ydiff = np.diff(curve_avg)
        peaks = signal.find_peaks(curve_avg)[0]
        ids0 = 3
        if any((peaks-ttp)<0):
            ids01 = peaks[(peaks-ttp)<0][-1]
            if ids01>ids0:
                ids0 = ids01
        
        steep = np.mean(abs(ydiff[ttp-4:ttp+4])) - np.std(abs(ydiff[ttp-4:ttp+4]))
        
        p = [n for n,v in enumerate(ydiff) if n<=ttp-2 and n>ids0 and v<0 and abs(v)>=steep]
        if any(p):
            ids02 = p[0]
            p2 = [n for n in p if abs(curve_avg[n]-np.mean(curve_avg[3:ids0+1]))>np.std(curve_avg) ]
            if any(p2):
                ids0 = p2[0]-2
            else:
                ids0 = ids02-1
        else:
            ids0 = np.where( ydiff[3:ttp-2]>np.mean(ydiff[ydiff<0]) )[0][-1]+3+1
        
        ids1 = len(curve_avg)-3
        if any((peaks-ttp)>0):
            ids11 = peaks[(peaks-ttp)>0][0]
            if ids11>ids1:
                ids1 = ids11
        p = [n for n,v in enumerate(ydiff) if n>=ttp+2 and n<ids1 and v>0 and abs(v)>=steep]
        if any(p):
            ids12 = p[-1]
            p2 = [n for n in p if abs(curve_avg[n]-np.mean(curve_avg[ids1-1:(len(curve_avg)-3)]))>np.std(curve_avg) ]
            if any(p2):
                ids1 = p2[0]+3
            else:
                ids1 = ids12+2
        else:
            ids1 = np.where( ydiff[ttp+2:]<np.mean(ydiff[ydiff>0]) )[0][0]+2+ttp
        return ttp,ids0,ids1
    else:
        return []


def curve_norm_wm_params(imdsc, wm, time, outfolder=[], out_plot=True, save_wm=True, p_print=None):
    
    logger.info('Extracting DSC normalization params')
    if not os.path.isdir(outfolder): os.makedirs(outfolder, exist_ok=True)
    if not p_print:
        p_print = outfolder
    
    curve_img,ttpm = check_mean_perfusion(imdsc)
    with open(os.path.join(outfolder,'dsc_avg.txt'), 'w+') as txtfile:
        txtfile.write(','.join(["{:.2f}".format(t) for t in time]))
        txtfile.write('\n')
        txtfile.write(','.join([str(round(v,2)) for v in curve_img]))
        txtfile.write('\n')
    if out_plot:
        plt.ioff()
        fig = plt.figure()
        plt.plot(time,curve_img)
        plt.xlabel('Time [s]')
        plt.ylabel('DSC intensity')
        fig.savefig(os.path.join(p_print,'dsc_avg.png'), dpi = 300)
        plt.close(fig)
    if not ttpm:
        logger.error('Not able to retrieve TTP from AVG curve, check DSC for bolus passage')
        end_execution()
    ttp,ids0,ids1 = get_curve_peaks(curve_img)
    
    curvearr_wm,new_mask = filter_curves(imdsc, wm, time, use_bolus_det=0)
    
    curve_avg_wm = np.mean(curvearr_wm,0)
    with open(os.path.join(outfolder,'wm_avg.txt'), 'w+') as txtfile:
        txtfile.write(','.join(["{:.2f}".format(t) for t in time]))
        txtfile.write('\n')
        txtfile.write(','.join([str(round(v,2)) for v in curve_avg_wm]))
        txtfile.write('\n')

    try:
        ttp_wm,ids0_wm,ids1_wm = get_curve_peaks(curve_avg_wm)
    except ValueError:
        logger.error('Not able to retrieve TTP from WM curve')
        logger.warning('Using AVG curve')
        ttp_wm,ids0_wm,ids1_wm = ttp,ids0,ids1
        curve_avg_wm = curve_img
    
    ph_wm = np.mean(curve_avg_wm[:ids0_wm])-curve_avg_wm[ttp_wm]
    if not abs(ttp_wm-ttp)<=1:
        logger.debug('TTP position might be wrong: %s %s',str(ttp_wm),str(ttp))
    
    time = np.array(time)
    
    ttp_time_wm = time/(time[ttp_wm]-time[ids0_wm])
    time_step = 0.2
    desired_time = np.arange(0,np.amax(ttp_time_wm)+time_step,time_step)
    ids0_norm = np.argmin(abs(desired_time-ttp_time_wm[ids0_wm]))
    
    curve_wm_norm = np.interp(desired_time, ttp_time_wm, curve_avg_wm)
    curve_wm_norm = (curve_wm_norm-np.mean(curve_wm_norm[:ids0_norm]))/ph_wm
    ttp_wm_norm,ids0_wm_norm,ids1_wm_norm = get_curve_peaks(curve_wm_norm)
    
    curvearr_wm = np.array(curvearr_wm)
    if save_wm:
        with open(os.path.join(outfolder,'wm_all.txt'), 'w+') as txtfile:
            txtfile.write(','.join(["{:.2f}".format(t) for t in time]))
            txtfile.write('\n')
            for l in range(curvearr_wm.shape[0]):
                txtfile.write(','.join([str(int(v)) for v in curvearr_wm[l,:]]))
                txtfile.write('\n')
    with open(os.path.join(outfolder,'info.txt'), 'w+') as txtfile:
        txtfile.write(','.join(['peak_height_wm','y_offset_wm','s0_ttp_wm_time','ids0_wm_norm','ttp_wm_norm','ids1_wm_norm','ids0_img','ttp_img','ids1_img', 'ids0_wm','ttp_wm','ids1_wm']))
        txtfile.write('\n')
        txtfile.write(','.join([str(round(ph_wm,2)), str(round(np.mean(curve_avg_wm[:ids0_wm]),2)), str(time[ttp_wm]-time[ids0_wm]), str(ids0_wm_norm), str(ttp_wm_norm), str(ids1_wm_norm), str(ids0), str(ttp), str(ids1), str(ids0_wm), str(ttp_wm), str(ids1_wm)]))
        txtfile.write('\n')
    with open(os.path.join(outfolder,'wm_norm.txt'), 'w+') as txtfile:
        txtfile.write(','.join(["{:.2f}".format(t) for t in desired_time]))
        txtfile.write('\n')
        txtfile.write(','.join([str(round(v,2)) for v in curve_wm_norm]))
        txtfile.write('\n')
    
    if out_plot:
        plt.ioff()
        fig = plt.figure()
        plt.plot(desired_time, curve_wm_norm)
        plt.xlabel('Time relative to WM')
        plt.ylabel('DSC relative to WM')
        fig.savefig(os.path.join(p_print,'wm_norm.png'), dpi = 300)
        plt.close(fig)

    return os.path.join(outfolder,'info.txt'), os.path.join(outfolder,'wm_norm.txt')


def normalize_to_wm(p_info, imdsc, mask, time, p_normwm, output_path=[], out_plot=True, p_print=None):
    minlen=34
    logger.info('Normalizing DSC curves')
    if not output_path:
        output_path = os.path.dirname(p_info)
        output_path = os.path.join(output_path, 'curves_mask')
    outfolder = output_path
    if not os.path.isdir(outfolder): os.makedirs(outfolder, exist_ok=True)
    if not p_print:
        p_print = outfolder
    
    with open(p_info, 'r+') as txtfile:
        rows = txtfile.readlines()
    ph_wm, _, wm_s0ttp = [float(i) for i in rows[-1].split(',')[0:3]]
    
    with open(p_normwm, 'r+') as txtfile:
        rows = txtfile.readlines()
    curve_wm_norm = [float(i) for i in rows[-1].split(',')]
    norm_time = [float(i) for i in rows[0].split(',')]
    
    curvearr,_ = filter_curves(imdsc, mask, time, use_bolus_det=0)
    
    curve_avg = np.mean(curvearr,0)
    ttp_time_wm = np.array(time)/wm_s0ttp
    curve_norm = np.interp(norm_time, ttp_time_wm, curve_avg)
    try:
        ttp_norm, ids0_norm, ids1_norm = get_curve_peaks(curve_norm)
    except ValueError:
        logger.error('Not able to retrieve TTP from MASK curve')
        end_execution()
    
    curve_norm = (curve_norm-np.mean(curve_norm[:ids0_norm]))/ph_wm
    
    flag_fail = check_mask_curve(curve_norm[ttp_norm])
    if flag_fail:
        logger.warning('MASK curve higher or equal than WM curve. Results may be wrong. Check masks and curves')
        # end_execution()
    
    curvearr = np.array(curvearr)
    with open(os.path.join(outfolder,'all.txt'), 'w+') as txtfile:
        txtfile.write(','.join(["{:.2f}".format(t) for t in time]))
        txtfile.write('\n')
        for l in range(curvearr.shape[0]):
            txtfile.write(','.join([str(int(v)) for v in curvearr[l,:]]))
            txtfile.write('\n')
    
    with open(os.path.join(outfolder,'info.txt'), 'w+') as txtfile:
        txtfile.write(','.join(['ids0_norm','ttp_norm','ids1_norm','y_offset']))
        txtfile.write('\n')
        txtfile.write(','.join([str(ids0_norm), str(ttp_norm), str(ids1_norm), str(round(np.mean(curve_avg[:ids0_norm]),2)) ]))
        txtfile.write('\n')
    curve_norm0 = curve_norm-curve_norm[ids0_norm]
    with open(os.path.join(outfolder,'norm_avg.txt'), 'w+') as txtfile:
        txtfile.write(','.join(["{:.2f}".format(t) for t in norm_time]))
        txtfile.write('\n')
        txtfile.write(','.join([str(round(v,2)) for v in curve_norm]))
        txtfile.write('\n')
    with open(os.path.join(outfolder,'norm_avg_no_tail.txt'), 'w+') as txtfile:
        txtfile.write(','.join(["{:.2f}".format(t) for t in norm_time[:-ids0_norm]]))
        txtfile.write('\n')
        txtfile.write(','.join([str(round(v,2)) for v in curve_norm0[ids0_norm:]]))
        txtfile.write('\n')
    
    if out_plot:
        fig = plt.figure()
        plt.plot(norm_time,curve_norm, label='Lesion', color='tab:orange')
        plt.plot(norm_time, curve_wm_norm, label='WM', color='tab:blue')
        plt.legend(loc='lower right')
        plt.xlabel('Time relative to WM TTP')
        plt.ylabel('DSC relative to WM PH')
        fig.savefig(os.path.join(p_print,'norm.png'), dpi = 300)
        plt.close(fig)
    if len(norm_time)<minlen:
        norm_time = [0.2*v for v in range(minlen)]
    with open(os.path.join(outfolder,'all_norm.txt'), 'w') as txtfile:
        txtfile.write(','.join(["{:.2f}".format(t) for t in norm_time]))
        txtfile.write('\n')
        for curve in curvearr:
            curve_norm = np.interp(norm_time, ttp_time_wm, curve)
            try:
                outp = get_curve_peaks(curve_norm)
            except IndexError:
                outp = []
            if outp:
                curve_norm = (curve_norm-curve_norm[(outp[1]-1)])/ph_wm
                curve_norm1 = curve_norm[(outp[1]-1):]
            else:
                curve_norm = (curve_norm-curve_norm[(ids0_norm-1)])/ph_wm
                curve_norm1 = curve_norm[(ids0_norm-1):]
            if len(curve_norm1)<len(norm_time):
                curve_norm[:len(curve_norm1)] = curve_norm1
                curve_norm[len(curve_norm1):] = np.repeat(np.mean(curve_norm1[-5:]), len(curve_norm)-len(curve_norm1))
            else:
                curve_norm = curve_norm1
            if len(curve_norm)<minlen:
                curve_norm = np.concatenate([curve_norm, np.repeat(np.mean(curve_norm[-5:]), minlen-len(curve_norm))],axis=0)
            txtfile.write(','.join([str(round(c,2)) for c in curve_norm]))
            txtfile.write('\n')
    return os.path.join(outfolder,'all_norm.txt')


def extract_norm_curves(p_dsc, p_mask, p_wm, output_path=[], out_plot=True, in_slicer_shape=True, p_print=None):
    logger.info('DSC curve processing')
    if not output_path:
        output_path = os.path.dirname(p_dsc)
    if not p_print:
        p_print = output_path
    outfolder = os.path.join(output_path, 'curves_mask')
    wmfolder = os.path.join(output_path, 'curves_norm_info')
    if not os.path.isdir(outfolder): os.makedirs(outfolder, exist_ok=True)
    if not os.path.isdir(wmfolder): os.makedirs(wmfolder, exist_ok=True)
    
    imdsc,hdsc = nrrd.read(p_dsc)
    if in_slicer_shape:
        imdsc = imdsc.transpose(1,2,3,0)
    mask,_ = nrrd.read(p_mask)
    mask[mask>0] = 1
    if np.sum(mask>0)<10:
        logger.error('Not enough enhancing region segmented!')
        logger.handlers.clear()
        logging.shutdown()
        end_execution()
    wm,_ = nrrd.read(p_wm)
    wm[wm>0] = 1
    if np.sum(wm>0)<10:
        logger.error('Not enough WM region segmented!')
        logger.handlers.clear()
        logging.shutdown()
        end_execution()
    
    if hdsc['MultiVolume.FrameIdentifyingDICOMTagUnits']=='ms':
        time = [int(float(v))/1000 for v in hdsc['MultiVolume.FrameLabels'].split(',')]
    elif hdsc['MultiVolume.FrameIdentifyingDICOMTagUnits']=='s':
        time = [int(float(v)) for v in hdsc['MultiVolume.FrameLabels'].split(',')]
    else:
        logger.info('Unkown time frame units')
    
    p_info, p_normwm = curve_norm_wm_params(imdsc, wm, time, outfolder=wmfolder, out_plot=out_plot, save_wm=True, p_print=p_print)
    p_curves = normalize_to_wm(p_info, imdsc, mask, time, p_normwm, output_path=outfolder, out_plot=out_plot, p_print=p_print)
    return p_curves, p_info

