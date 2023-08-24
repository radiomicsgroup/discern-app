# -*- coding: utf-8 -*-
"""
classifiers
"""

from app.inout.inout import write_hist, write_map
from app.utils import end_execution
from pandas import read_csv
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)


def call_model(data, p_model, p_scaling):
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.models import load_model, Model
    from pickle import load

    model = load_model(p_model, compile=False)
    with open(p_scaling, 'rb') as f:
        scaler = load(f)
    datasc = scaler.transform(data)
    x_test = np.expand_dims(datasc, axis=-1)
    probs = model.predict([x_test, x_test, x_test])
    classes = np.argmax(probs, axis=1)
    return probs, classes


def write_results(probs, classes, p_output):
    os.makedirs(p_output, exist_ok=True)
    with open(os.path.join(p_output, 'probs.txt'), 'w') as txtfile:
        for p in probs:
            txtfile.write(','.join([str(v) for v in p]))
            txtfile.write('\n')
    with open(os.path.join(p_output, 'classes.txt'), 'w') as txtfile:
        txtfile.write('\n'.join([str(v) for v in classes]))
        txtfile.write('\n')
    with open(os.path.join(p_output, 'class_score.txt'), 'w') as txtfile:
        txtfile.write('Label 0:'+str(round(sum(classes == 0)/len(classes), 3)))
        txtfile.write('\n')
        txtfile.write('Label 1:'+str(round(sum(classes == 1)/len(classes), 3)))
        txtfile.write('\n')
    

def data_class_task(p_data, p_output, p_model1, p_model2, p_lesion, threshold1=60, threshold2=84, p_t1r=[], printout=False, p_print=None):
    logger.info('Launching classifier')
    minlen = 34
    warnConfTh = 0.2
    confTh = 0.9
    th1 = threshold1/100
    th2 = threshold2/100
    data = read_csv(p_data, header=None, sep=',', decimal='.')
    if data.shape[-1] < minlen:
        logger.error('Insufficient DSC time points: %d', data.shape[-1])
        end_execution()
    data = data.iloc[1:, :minlen]
    logger.debug('Running Lymphoma classifier')
    probs, classes = call_model(data, p_model=os.path.join(p_model1, 'model_trained.h5'), p_scaling=os.path.join(p_model1, 'scaler_training.pckl'))
    write_results(probs, classes, os.path.join(p_output, 'class_lym.tmp'))
    write_map(probs, classes, p_lesion, os.path.join(p_output, 'class_lym.tmp'), printout, p_t1r, p_print=os.path.join(p_print, 'probmap_lym'), cmapname='cool', labels=['Non-lymphoma', 'Lymphoma'])
    write_hist(probs[:, 1],os.path.join(p_output,'results',"probhist.png"))
    ncl1 = sum(probs[:, 1] >= confTh)
    ncl0 = sum(probs[:, 0] >= confTh)
    ncl = ncl0+ncl1
    probrel = ncl1/(ncl)
    if ncl/len(probs) < warnConfTh:
        logger.warning('Only %s %% of high confident voxels. Proceed with caution!', round(ncl/len(probs)*100))
    if not ncl:
        lym = 1
        resultxt = 'CNN 1: No confident voxels found for prediction!'
    else:
        logger.debug('Voxels used for classification: %s (%s %%), of which class Lymphoma: %s (%s %%),', ncl, round(ncl/len(probs)*100), ncl1, round(probrel*100))
        lym = probrel > th1
        logger.debug('Lymphoma: %s', lym)
    
    met = None
    if not lym:
        logger.debug('Running Metastasis classifier')
        probs, classes = call_model(data, p_model=os.path.join(p_model2, 'model_trained.h5'), p_scaling=os.path.join(p_model2, 'scaler_training.pckl'))
        write_results(probs, classes, os.path.join(p_output, 'class_met.tmp'))
        write_map(probs, classes, p_lesion, os.path.join(p_output, 'class_met.tmp'), printout, p_t1r, p_print=os.path.join(p_print, 'probmap_gbmet'), cmapname='autumn', labels=['Glioblastoma', 'Metastasis'])
        write_hist(probs[:, 1],os.path.join(p_output,'results',"probhist.png"))
        ncl1 = sum(probs[:, 1] >= confTh)
        ncl0 = sum(probs[:, 0] >= confTh)
        ncl = ncl0+ncl1
        probrel2 = ncl1/(ncl)
        if ncl/len(probs) < warnConfTh:
            logger.warning('Only %s %% of high confident voxels. Proceed with caution!', round(ncl/len(probs)*100))
        if not ncl:
            met = 1
            resultxt = 'CNN 2: No confident voxels found for prediction!'
        else:
            logger.debug('Voxels used for classification: %s (%s %%), of which class Metastasis: %s (%s %%),', ncl, round(ncl/len(probs)*100), ncl1, round(probrel2*100))
            met = probrel2 > th2
            logger.debug('Metastasis: %s', met)

    if ncl:
        if lym:
            resultclass = 'Lymphoma'
            scoreclass = probrel*100
        else:
            scoreclass = probrel2*100
            if met:
                resultclass = 'Metastasis'
            else:
                resultclass = 'Glioblastoma'
                scoreclass = 100-scoreclass
        resultxt = 'CNN: {result} with proportion: {score:.2f} %'.format(result=resultclass, score=scoreclass)
    logger.info(resultxt)
    with open(os.path.join(p_print, 'result_message.txt'), 'w') as resfile:
        resfile.write(resultxt)
    return lym, met


def class_rcbv(rcbv, p_output, thlym=2, thgb=2.13):
    logger.debug('Classifying by rCBV')
    if rcbv < thlym:
        resultxt = 'Lymphoma'
    elif rcbv > thgb:
        resultxt = 'Glioblastoma'
    else:
        resultxt = 'Metastasis'
    logger.debug('rCBV: %s with rCBV: %s', resultxt, rcbv)
    with open(os.path.join(p_output, 'result_rcbv.txt'), 'w') as resfile:
        resfile.write(','.join([resultxt, str(rcbv)]))


def class_psr(psr, p_output, thlym=1.34, thgb=0.98):
    logger.debug('Classifying by PSR')
    if psr >= thlym:
        resultxt = 'Lymphoma'
    elif psr < thgb:
        resultxt = 'Metastasis'
    else:
        resultxt = 'Glioblastoma'
    logger.debug('PSR: %s with PSR: %s', resultxt, psr)
    with open(os.path.join(p_output, 'result_psr.txt'), 'w') as resfile:
        resfile.write(','.join([resultxt, str(psr)]))


def class_logreg(rcbv, psr, p_output, p_model1, p_model2):
    logger.debug('Classifying by logreg')
    from pickle import load
    with open(os.path.join(p_model1, 'logreg_dsc_lym.pckl'), 'rb') as f:
        modellym = load(f)
    pred = modellym.predict(([psr, rcbv],))[0]
    if pred == 'Other':
        with open(os.path.join(p_model2, 'logreg_dsc_gbmet.pckl'), 'rb') as f:
            modelgbmet = load(f)
        pred = modelgbmet.predict(([psr, rcbv],))[0]
    logger.debug('Case classified by logreg as: %s', pred)
    with open(os.path.join(p_output, 'result_logreg.txt'), 'w') as resfile:
        resfile.write(pred)
