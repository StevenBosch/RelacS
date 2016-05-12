import sys
import h5py
import numpy as np
import os
import os.path
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("usage:", sys.argv[0], "<name.hdf5>")
        sys.exit(0)

    filename = sys.argv[1]
    filepointer = h5py.File(filename, 'r')

    energy = filepointer['energy']
    coch = energy.value
    
