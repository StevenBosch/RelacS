import sys
import os

label_dir = os.path.join(os.getcwd(), '../labeling')
sys.path.insert(0, label_dir)

import windows_to_images

import matplotlib.pyplot as plt

import pickle

import numpy as np
import theano

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD




import math

def build_data(train_split = 0.9) :

    import h5py
    
    #load the windows from file
    windows = pickle.load(open(os.path.join(label_dir, 'labels.pickle'), 'rb'))

    #cut out the actual cochleogram fragments 
    # and generate a list of corresponding flags (stressfull, relaxing, sudden)
    codedir, dummy= os.path.split(os.getcwd())
    relacsdir, dummy = os.path.split(codedir)
    hdf5_path = os.path.join(relacsdir, 'sound_files/hdf5')
    X_train, Y_train, X_test, Y_test = windows_to_images.to_image_data_file_split(windows, train_split, hdf5_path)

    xshape = (X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    yshape = (X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    X_train = np.reshape(X_train, xshape)
    X_test = np.reshape(X_test, yshape)

    print 'Saving data in:'
    print 'X_train...'
    with open('xtrain2.np', 'wb') as f:
        np.save(f, X_train)
    print 'Y_train...'
    with open('ytrain2.np', 'wb') as f:
        np.save(f, Y_train)
    print 'X_test...'
    with open('xtest2.np', 'wb') as f:
        np.save(f, X_test)
    print 'Y_test...'
    with open('ytest2.np', 'wb') as f:
        np.save(f, Y_test)
    print 'Data saved.'
    return X_train, Y_train, X_test, Y_test

def load_data() :
    print 'Loading data from:'
    print 'X_train...'
    with open('xtrain.np', 'rb') as f:
        X_train= np.load(f)
    print 'Y_train...'
    with open('ytrain.np', 'rb') as f:
        Y_train= np.load(f)
    print 'X_test...'
    with open('xtest.np', 'rb') as f:
        X_test= np.load(f)
    print 'Y_test...'
    with open('ytest.np', 'rb') as f:
        Y_test= np.load(f)
    print 'Data loaded.'
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

    # model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(64, 3, 3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
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

def same_number_of_idxs(x, y):
    no_stress_idxs = np.asarray([], dtype = int)
    stress_idxs = np.asarray([], dtype = int)
    for i in range(y.shape[0]):
       if y[i] == 1:
           stress_idxs = np.append(stress_idxs, i)
       else:
           no_stress_idxs = np.append(no_stress_idxs, i)

    idxs = np.random.choice(no_stress_idxs, stress_idxs.shape[0], replace = False)

    idxs = np.append(idxs, stress_idxs)

    y = y[idxs]
    x = x[idxs]

    return x,y

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = build_data()
    # X_train, Y_train, X_test, Y_test = load_data()
    Y_train = Y_train[:, 0]
    Y_test = Y_test[:, 0]

    X_train -= 86
    X_train /= 255
    X_test  -= 86
    X_test  /= 255
    
    plt.hist(X_train.flatten(), bins = 200)
    plt.show()

    plt.hist(X_test.flatten(), bins = 200)
    plt.show()

    print np.max(X_train)
    print np.max(X_test)
    print np.mean(X_train)
    print np.mean(X_test)

    X_train, Y_train = same_number_of_idxs(X_train, Y_train)
    X_test,  Y_test  = same_number_of_idxs(X_test,  Y_test)

    print sum(Y_train)/len(Y_train)
    print sum(Y_test)/len(Y_test)

    model = build_empty_model(X_train.shape, Y_train.shape)

    model.load_weights('weights_3') 
    
    model.fit(X_train, Y_train, batch_size = 16, nb_epoch = 10, validation_data= (X_test, Y_test))


    # loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=16) 
    
    # print loss_and_metrics

    print 'predicting output'
    output = model.predict(X_test)

    
    thrs = [0.2, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for thr in thrs:
        cp = 0
        cn = 0
        fp = 0
        fn = 0

        for it in range(len(output)): 
            if (output[it] >= thr) == Y_test[it]:
                if output[it] >= thr:
                    cp += 1
                else :
                    cn += 1
            else :
                if output[it] >= thr:
                    fp += 1
                else :
                    fn += 1
        print 'thr: ' + str(thr),
        print 'acc: ' + str(round((cp+cn) / float(len(output)),2)),
        print '  %fn: ' + str(round((fn) / float(len(output)),2)),
        print '  %fp: ' + str(round((fp) / float(len(output)),2)),
        print '  %cn: ' + str(round((cn) / float(len(output)),2)),
        print '  %cp: ' + str(round((cp) / float(len(output)),2))

    filedir = os.path.join(os.getcwd())
    filename = os.path.join(filedir, 'weights_4')
    model.save_weights(filename)

# Die vorige was met / 255 en -mean

