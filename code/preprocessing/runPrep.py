import preprocessing as p
import sys, glob, shutil
import cv2, h5py, numpy as np
import os, matplotlib.pyplot as plt

# Change search directory for modules to include pycpsp in parent
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import pycpsp.files as files
import pycpsp.bgmodel.config as bgconfig
import pycpsp.bgmodel.bgmodel as bgmodel

def prepFile(f):
    print "file: " + f
    if os.path.isfile(f):
        # Read information from the hdf5
        signals = files.signalsFromHDF5(f)

        # Original signal
        original = signals['energy'][:,:]

        # Get the zero/infinity columns out
        original = np.transpose(original)
        zeroColLeft = []
        zeroColRight = []
        while max(original[0]) <= 0:
            zeroColLeft.append(original[0])
            original = original[1:]
        while max(original[-1]) <= 0:
            zeroColRight.append(original[-1])
            original = original[:-2]
        original = np.transpose(original)

        # preprocess signal
        morphology = p.dilate(p.erode(original))

        # Put the zeros/infinities back
        if len(zeroColLeft) > 0 or len(zeroColRight) > 0:
            if len(zeroColLeft) > 0:
                morphology = np.column_stack((np.transpose(zeroColLeft), morphology))
            if len(zeroColRight) > 0:
                morphology = np.column_stack((morphology, np.transpose(zeroColRight)))
        
        
        # Store the preprocessed signal
        if 'morphology' in signals:
            signals['morphology'] = morphology
        else:
            with h5py.File(f, 'r+') as filepointer:
                filepointer.create_dataset('morphology', data=morphology)

        # Foreground background
        definitions = [bgconfig.getDefaults(tau) for tau in bgconfig.tau(0.5, 4)]
        bgmodels = [bgmodel.BGModel(d['tau'], d,) for d in definitions]
        for model in bgmodels:
            response = model.calculate(original)
            # Put the zeros/infinities back
            if len(zeroColLeft) > 0 or len(zeroColRight) > 0:
                if len(zeroColLeft) > 0:
                    response = np.column_stack((np.transpose(zeroColLeft), response))
                if len(zeroColRight) > 0:
                    response = np.column_stack((response, np.transpose(zeroColRight)))
            # np.save(prepDir + 'tau {}'.format(model.name), response)
            if 'tau {}'.format(model.name) in signals:
                signals['tau {}'.format(model.name)] = response
            else:
                with h5py.File(f, 'r+') as filepointer:
                    filepointer.create_dataset('tau {}'.format(model.name), data=response)

def prepDir(hdf5Dir):
    filelist = glob.glob(hdf5Dir + '*')
    for f in filelist:
        prepFile(f)
