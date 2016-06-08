
import sys
import os

import numpy as np


label_dir = os.path.join(os.getcwd(), '../CNN')
sys.path.insert(0, label_dir)

import cnn

import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


def get_len(window) :
    window = np.reshape(window, window.shape[1:])
    window = np.sum(window, axis = 1)
        
    return np.sum(np.absolute(np.diff(window)))

def get_maxminmin(window) :
    window = np.reshape(window, window.shape[1:])
    window = np.sum(window, axis = 1)
        
    return max(window) - min(window)
    
    

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = cnn.load_data()

    stressful = []
    not_stressful = []
    for idx, win in enumerate(X_train[:,:,:,30:]) :
        length = get_len(win)
        maxminmin= get_maxminmin(win) 
        if Y_train[idx, 0] :
            stressful.append(length)
        else :
            not_stressful.append(length)

    plt.figure()
    plt.subplot(311)
    hist1 = np.histogram(stressful, bins = 1000, range = (0,90000))[0] / float(len(stressful))
    plt.plot(hist1)

    plt.subplot(312)
    hist2 = np.histogram(not_stressful, bins = 1000, range = (0,90000))[0] / float(len(not_stressful))
    plt.plot(hist2)

    plt.subplot(313)
    plt.plot(hist1 - hist2)

    plt.show()
