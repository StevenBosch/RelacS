import preprocessing as p
import sys, glob, shutil
import cv2, h5py, numpy as np
import os, matplotlib.pyplot as plt

# Change search directory for modules to include pycpsp in parent
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import pycpsp.plot as plot
import pycpsp.files as files
import pycpsp.bgmodel.config as bgconfig
import pycpsp.bgmodel.bgmodel as bgmodel

filelist = glob.glob('../../sound_files/hdf5/*')

if os.path.exists('../../sound_files/preprocessed'):
    shutil.rmtree('../../sound_files/preprocessed')
os.makedirs('../../sound_files/preprocessed')

for f in filelist:
    print "file: " + f
    if os.path.isfile(f):
        # Read information from the hdf5
        filename = os.path.basename(f)
        signals = files.signalsFromHDF5(f)
        prepDir = '../../sound_files/preprocessed/' + filename + '/'
        os.makedirs(prepDir)

        # Original signal (44 because of the infinities in the first part)
        original = signals['energy'][:,44:]

        # preprocess signals
        eroded = p.erode(original)
        dilated = p.dilate(original)
        combined = p.dilate(p.erode(original))
        p.foregroundBackground(original, prepDir)

        # Save the signals
        np.save(prepDir + 'original', original)
        np.save(prepDir + 'eroded', eroded)
        np.save(prepDir + 'dilated', dilated)
        np.save(prepDir + 'combined', combined)
