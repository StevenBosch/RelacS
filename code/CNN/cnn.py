import sys
import os

label_dir = os.path.join(os.getcwd(), '../labeling')
sys.path.insert(0, label_dir)


import windows_to_images

import matplotlib.pyplot as plt

import pickle
import h5py

import numpy as np
import theano

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD




import math

def build_data() :

    #load the windows from file
    windows = pickle.load(open(os.path.join(label_dir, 'labels.pickle'), 'rb'))

    #cut out the actual cochleogram fragments 
    # and generate a list of corresponding flags (stressfull, relaxing, sudden)
    codedir, dummy= os.path.split(os.getcwd())
    relacsdir, dummy = os.path.split(codedir)
    hdf5_path = os.path.join(relacsdir, 'sound_files/hdf5')
    X_train, Y_train, X_test, Y_test = windows_to_images.to_image_data_file_split(windows, 0.75, hdf5_path)

    xshape = (X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    yshape = (X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    X_train = np.reshape(X_train, xshape)
    X_test = np.reshape(X_test, yshape)

    #print 'Saving data in data.pickle'
    #with open('data.pickle', 'wb') as f:
    #    pickle.dump((X_train, Y_train, X_test, Y_test), f)

    return X_train, Y_train, X_test, Y_test

def load_data() :
    print 'Loading data from data.pickle'
    with open('data.pickle', 'wr') as f:
        f.seek(0)
        (X_train, Y_train, X_test, Y_test) = pickle.load(f)
    return X_train, Y_train, X_test, Y_test


def build_empty_model(X_shape, Y_shape) :
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.

    model.add(Convolution2D(32, 3, 3, border_mode ='valid', input_shape= X_shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    if len(Y_shape) == 1:
        Y_shape = np.array([Y_shape, 1])
    model.add(Dense(Y_shape[1]))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer= sgd,
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = build_data() #doesn't work
    #X_train, Y_train, X_test, Y_test = load_data()
    Y_train = Y_train[:, 0]
    Y_test = Y_test[:, 0]
    
    #no_stress_idxs = np.array([])
    #stress_idxs = np.array([])
    #for i in range(Y_train.shape[0]):
    #    if Y_train[i] == 1:
    #        stress_idxs = np.append(stress_idxs, i)
    #    else:
    #        no_stress_idxs = np.append(no_stress_idxs, i)

    #print stress_idxs.shape
    #print no_stress_idxs.shape

    #idxs = np.random.choice(stress_idxs, no_stress_idxs.shape[0], replace = False)

    #np.append(idxs, no_stress_idxs)

    #Y_train = Y_train[idxs]
    #X_train = X_train[idxs]

    #print Y_train.shape
    #print X_train.shape

    model = build_empty_model(X_train.shape, Y_train.shape)

    model.fit(X_train, Y_train, batch_size = 16, nb_epoch = 10, validation_data= (X_test, Y_test))

    #model.load_weights('weights') 

    loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=16) 
    
    print loss_and_metrics

    print 'predicting output'
    output = model.predict(X_test)

    for o in output: 
        if o[0] != 1 or o[1] != 0 or o[2] != 0 :    
            print o

    filedir = os.path.join(os.getcwd())
    filename = os.path.join(filedir, 'weights')
    model.save_weights(filename)



