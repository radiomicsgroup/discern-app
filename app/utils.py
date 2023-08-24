# -*- coding: utf-8 -*-
"""
utilities
"""

import os
import traceback
import sys
import numpy as np
from subprocess import Popen, PIPE
import shlex

def end_execution():
    sys.exit()

class Tee(object):
    def __init__(self, filename, mode):
        self.file = open(filename, mode)
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

def listfolders(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

def listfiles(folder):
    return [d for d in os.listdir(folder) if not os.path.isdir(os.path.join(folder, d))]

def wrap_string(s,ch):
    news = ch+s+ch
    return news

def cast_to_type(img, newtype):
    if 'uint' in newtype:
        img[img<0] = 0
    img = np.array(img, dtype=newtype)
    return img

def get_exitcode_stdout_stderr(cmd):
    """
    Execute the external command and get its exitcode, stdout and stderr.
    """
    args = shlex.split(cmd)
    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    return exitcode, out, err

def create_json(p_output, filename, res, logfile=None):
    """Generate JSON output file

    Args:
        p_output (string): folder output
        filename (string): file name
        res (_type_): _description_
        logfile (list strings): List of path of log files. Defaults to None.
    """
    from lenden.dataexport import DataResults, DataExport, ResultType, ResultSizeType, DisplayType, DataLog
    results =[]

    results.append(DataResults(Description="Normalized average DSC curves",
                     Help="The DSC curves from the tumour are normalized to the white matter (WM) intensity peak and time to peak. The tumour curve should deviate from the WM (check documentation). If the tumour curve is above or similar to the WM curve, be wary of the results (the segmentation might be wrong).",
                     Type=ResultType.image, Path=os.path.join(p_output,"norm.png"), Size=ResultSizeType.normal, Format="%20f"))

    results.append(DataResults(Description="Reference image",
                     Help="This image can be used to check whether the result of the segmentation is satisfactory.",
                     Type=ResultType.images, Path=os.path.join(p_output,"ref_im"), Size=ResultSizeType.normal, Format="%20f"))

    results.append(DataResults(Description="Segmentation of enhancing tumour",
                     Help="This image shows the enhancing lesion segmentation overlayed over the reference image.",
                     Type=ResultType.images, Path=os.path.join(p_output,"seg_lesion"), Size=ResultSizeType.normal, Format="%20f"))

    results.append(DataResults(Description="Segmentation of white matter",
                     Help="This image shows the white matter segmentation overlayed over the reference image.",
                     Type=ResultType.images, Path=os.path.join(p_output,"seg_wm"), Size=ResultSizeType.normal, Format="%20f"))
    if res[0]:
        results.append(DataResults(Description="Probability map for lymphoma",
                        Help="This images shows the probability map overlayed over the reference image. Closer to 1 means higher similitude with a characteristic lymphoma perfusion curve.",
                        Type=ResultType.images, Path=os.path.join(p_output,"probmap_lym"), Size=ResultSizeType.normal, Format="%20f"))
    else:
        results.append(DataResults(Description="Probability map for metastasis",
                        Help="This images shows the probability map overlayed over the reference image. Closer to 1 means higher similitude with a characteristic metastasis perfusion curve.",
                        Type=ResultType.images, Path=os.path.join(p_output,"probmap_gbmet"), Size=ResultSizeType.normal, Format="%20f"))
    
    results.append(DataResults(Description="Histogram of voxel probabilities",
                     Help="This image shows the distribution of probabilities found within the segmentation.",
                     Type=ResultType.image, Path=os.path.join(p_output,"probhist.png"), Size=ResultSizeType.normal, Format="%20f"))

    results.append(DataResults(Description="Result of CNN classification",
                        Help="The result of the CNN shows the proportion of found voxels for the tumour type given.",
                        Type=ResultType.str, Path=os.path.join(p_output,"result_message.txt"), Size=ResultSizeType.normal, Format="%20f"))

    LogsObj = None
    if logfile  is not None:
        listlogs = []
        LogsObj = DataLog("Default Logs", listlogs +  logfile)
    cleanpatterns = ['*.tmp']
    data = DataExport(Version=0.1, HasError=False,
                            DisplayType=DisplayType.grid, Results=results, CleanPatterns=cleanpatterns, Logs=LogsObj)
    with open(os.path.join(p_output,filename),'w') as j:
        j.write(data.getJSON())
