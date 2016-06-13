import os
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np

def fun(wave) :

    filename = os.path.join(os.getcwd(), 'logsweeps/' + wave + '.wav.2.hdf5')

    filepointer = h5py.File(filename, 'r')  
    plt.imshow(np.flipud(filepointer['energy']))
    plt.show()

if __name__ == '__main__':
    fun('sawtooth')