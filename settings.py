# -*- encoding: utf-8 -*-
"""
neuro-app settings
"""

import os
from decouple import config

LOG_FOLDER= config('LOG_FOLDER', default="./logs/" )

SLICER_EXEC = config('SLICER_EXEC', default="./programs/Slicer-4.8.1-linux-amd64/Slicer" )
EMSEGMENTCOMMANDLINE = config('EMSEGMENTCOMMANDLINE', default="./programs/Slicer-4.8.1-linux-amd64/lib/Slicer-4.8/cli-modules/EMSegmentCommandLine" )
MRML= config('MRML', default="./programs/Slicer-4.8.1-linux-amd64/share/Slicer-4.8/qt-loadable-modules/EMSegment/Tasks/MRI-Human-Brain-Hemisphere.mrml" )

DCM2NIIX=config('DCM2NIIX', default="./programs/dcm2niix" )

MODEL1 = config('MODEL1', default="./models/lym_current" )
MODEL2 = config('MODEL2', default="./models/gbmet_current" )

DEBUG = config('DEBUG', default=False, cast=bool)
