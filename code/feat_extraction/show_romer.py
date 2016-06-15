
import sys
import os
import pickle

import numpy as np

label_dir = os.path.join(os.getcwd(), '../CNN')
sys.path.insert(0, label_dir)

import cnn

import matplotlib.pyplot as plt
import math
from scipy import interpolate
from scipy import stats
from scipy.ndimage.filters import gaussian_filter1d
def show_romer() :
    import h5py
    codedir, dummy = os.path.split(os.getcwd())
    relacsdir, dummy = os.path.split(codedir)
    hdf5_path = os.path.join(relacsdir, 'sound_files/hdf5')

    filename = os.path.join(hdf5_path, 'S.M.Romer_2016-04-25T08:20:59.000Z.wav.2.hdf5')
    filepointer = h5py.File(filename, 'r')  

    plt.imshow(filepointer['energy'][:,:1000])
    plt.show()
    plt.imshow(filepointer['energy'][:,1000:2000])
    plt.show()
    plt.imshow(filepointer['energy'][:,2000:3000])
    plt.show()
    plt.imshow(filepointer['energy'][:,10000:])
    plt.show()
    quit()

if __name__ == '__main__':
    show_romer()
