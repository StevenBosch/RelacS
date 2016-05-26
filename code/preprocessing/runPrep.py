import preprocessing as p
import sys, glob, shutil
import cv2, h5py, numpy as np
import os, matplotlib.pyplot as plt

# Change search directory for modules to include pycpsp in parent
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# import pycpsp.plot as plot
import pycpsp.files as files
import pycpsp.bgmodel.config as bgconfig
import pycpsp.bgmodel.bgmodel as bgmodel

def runPrep(hdf5Dir):
    #prepDir = hdf5Dir + '../preprocessed'
    #if os.path.exists(prepDir):
    #    shutil.rmtree(prepDir)
    #os.makedirs(prepDir)

    filelist = glob.glob(hdf5Dir + '*')
    for f in filelist:
        print "file: " + f
        if os.path.isfile(f):
            # Read information from the hdf5
            filepointer = h5py.File(f, 'r+')
            signals = files.signalsFromHDF5(f)
            # filename = os.path.basename(f)
            # os.makedirs(prepDir + filename + '/')

            # Original signal (44 because of the infinities in the first part)
            original = signals['energy'][:,44:]

            # preprocess signal
            morphology = p.dilate(p.erode(original))

            if 'morphology' in signals:
                signals['morphology'] = morphology
            else:
                filepointer.create_dataset('morphology', data=morphology)

            # Foreground background
            definitions = [bgconfig.getDefaults(tau) for tau in bgconfig.tau(0.5, 4)]
            bgmodels = [bgmodel.BGModel(d['tau'], d,) for d in definitions]
            for model in bgmodels:
                response = model.calculate(original)
                # np.save(prepDir + 'tau {}'.format(model.name), response)
                if 'tau {}'.format(model.name) in signals:
                    signals['tau {}'.format(model.name)] = response
                else:
                    filepointer.create_dataset('tau {}'.format(model.name), data=response)

            '''
            # Save the signals
            np.save(prepDir + 'original', original)
            np.save(prepDir + 'eroded', eroded)
            np.save(prepDir + 'dilated', dilated)
            np.save(prepDir + 'combined', combined)
            '''
