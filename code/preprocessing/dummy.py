import sys, glob, shutil
import h5py, numpy as np
import os, matplotlib.pyplot as plt

# Change search directory for modules to include pycpsp in parent
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import pycpsp.files as files
import pycpsp.bgmodel.config as bgconfig
import pycpsp.bgmodel.bgmodel as bgmodel

if __name__ == '__main__':
    f = sys.argv[1]
    if os.path.isfile(f):
        # Read information from the hdf5
        signals = files.signalsFromHDF5(f)

        # Original signal
        original = signals['energy'][:,:]
        print original.shape
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
        print original.shape
        
        # Put the zeros/infinities back
        if len(zeroColLeft) > 0 or len(zeroColRight) > 0:
            if len(zeroColLeft) > 0:
                morphology = np.column_stack((np.transpose(zeroColLeft), original))
            if len(zeroColRight) > 0:
                morphology = np.column_stack((original, np.transpose(zeroColRight)))
        print morphology.shape