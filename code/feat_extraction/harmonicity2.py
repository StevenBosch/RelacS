import os, sys, glob
import h5py
import matplotlib.pyplot as plt
import numpy as np

def load_wave(wavetype) :    
    sweep_file = os.path.join(os.getcwd(), 'logsweeps/' + wavetype + '.wav.2.hdf5')
    #load the windows from file
    filepointer = h5py.File(sweep_file, 'r')
    plt.imshow(filepointer['energy'][:,170:1160])
    plt.show()

def fun(wave) :

    filename = os.path.join(os.getcwd(), 'logsweeps/' + wave + '.wav.2.hdf5')

    filepointer = h5py.File(filename, 'r')  
    plt.imshow((filepointer['energy']))
    plt.show()

if __name__ == '__main__':
    load_wave('sawtooth')
    load_wave('block')
    load_wave('triangle')
    load_wave('sine')