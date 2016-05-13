import preprocessing as p
import sys, glob
import cv2, h5py, numpy as np
import os, matplotlib.pyplot as plt

# Change search directory for modules to include pycpsp in parent
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import pycpsp.plot as plot
import pycpsp.files as files
import pycpsp.bgmodel.config as bgconfig
import pycpsp.bgmodel.bgmodel as bgmodel

filelist = glob.glob('../../sound_files/hdf5/*')

if not os.path.exists('../../sound_files/preprocessed'):
    os.makedirs('../../sound_files/preprocessed')

for f in filelist:
    print "file: " + f
    if os.path.isfile(f):
        # Read information from the hdf5
        filepointer = h5py.File(f, 'r')
        filename = os.path.basename(f)
        signals = files.signalsFromHDF5(f)
        metadata = files.metadataFromHDF5(f)

        if not os.path.exists('../../sound_files/preprocessed/' + filename):
            os.makedirs('../../sound_files/preprocessed/' + filename)

        # Original image
        original = signals['energy'][:, metadata.get('starttime 0'):]
        refsignal = plot.imgmask(plot.imscale(signals['energy'],[0,60]), [20,80])

        # preprocess images
        eroded = p.erode(refsignal)
        dilated = p.dilate(refsignal)
        combined = p.dilate(p.erode(refsignal))
        p.foregroundBackground(refsignal, filename, metadata, True)

        # Save the plots
        p.savePlot(refsignal, 'refsignal', filename, metadata, run = True)
        p.savePlot(original, 'original', filename, metadata, run = True)
        p.savePlot(eroded, 'eroded', filename, metadata, run = True)
        p.savePlot(dilated, 'dilated', filename, metadata, run = True)
        p.savePlot(combined, 'combined', filename, metadata, run = True)
