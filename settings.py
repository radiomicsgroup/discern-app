# -*- encoding: utf-8 -*-

import os
from decouple import config

"""
neuro-app settings, example in Linux
"""

# LOG_FOLDER= config('LOG_FOLDER', default="./logs/" )

# SLICER_ROOT = config('SLICER_ROOT', default="./programs/Slicer-4.8.1-linux-amd64" )
# DCM2NIIX=config('DCM2NIIX', default="./programs/dcm2niix/dcm2niix" )

# SLICER_EXEC = config('SLICER_EXEC', default=os.path.join(SLICER_ROOT,"Slicer"))
# EMSEGMENTCOMMANDLINE = config('EMSEGMENTCOMMANDLINE', default=os.path.join(SLICER_ROOT,"lib","Slicer-4.8","cli-modules","EMSegmentCommandLine"))
# MRML= config('MRML', default=os.path.join(SLICER_ROOT,"share","Slicer-4.8","qt-loadable-modules","EMSegment","Tasks","MRI-Human-Brain-Hemisphere.mrml"))

# MODEL1 = config('MODEL1', default="./models/lym_current" )
# MODEL2 = config('MODEL2', default="./models/gbmet_current" )

# DEBUG = config('DEBUG', default=False, cast=bool)


"""
neuro-app settings, example in Windows
"""

LOG_FOLDER= config('LOG_FOLDER', default=".\logs" )

SLICER_ROOT = config('SLICER_ROOT', default="C:\Program Files\Slicer4.8.1")
DCM2NIIX=config('DCM2NIIX', default='"'+os.path.normpath("C:\SOFT\dcm2niix\dcm2niix.exe")+'"')

SLICER_EXEC = config('SLICER_EXEC', default='"'+os.path.join(SLICER_ROOT,"Slicer.exe")+'"')
EMSEGMENTCOMMANDLINE = config('EMSEGMENTCOMMANDLINE', default='"'+ os.path.join(SLICER_ROOT,"lib","Slicer-4.8","cli-modules","EMSegmentCommandLine.exe")+'"')
MRML= config('MRML', default=os.path.join(SLICER_ROOT,"share","Slicer-4.8","qt-loadable-modules","EMSegment","Tasks","MRI-Human-Brain-Hemisphere.mrml"))

MODEL1 = config('MODEL1', default=".\models\lym_current" )
MODEL2 = config('MODEL2', default=".\models\gbmet_current" )

DEBUG = config('DEBUG', default=False, cast=bool)

