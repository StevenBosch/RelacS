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

def build_data(imageType = 'energy', key = '_', train_split = 0.9) :

    import h5py
    
    #load the windows from file
    windows = pickle.load(open(os.path.join(label_dir, 'labels.pickle'), 'rb'))

    #cut out the actual cochleogram fragments 
    # and generate a list of corresponding flags (stressfull, relaxing, sudden)
    codedir, dummy = os.path.split(os.getcwd())
    relacsdir, dummy = os.path.split(codedir)
    hdf5_path = os.path.join(relacsdir, 'sound_files/hdf5')
    X_train, Y_train, X_test, Y_test = windows_to_images.to_image_data_file_split(windows, train_split, hdf5_path, imageType)

    xshape = (X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    yshape = (X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    X_train = np.reshape(X_train, xshape)
    X_test = np.reshape(X_test, yshape)

    data_dir = os.path.join(os.getcwd(), '../data')
    print 'Saving data of:'
    print 'X_train...'
    with open(os.path.join(data_dir, 'xtrain' + key + '.np'), 'wb') as f:
        np.save(f, X_train)
    print 'Y_train...'
    with open(os.path.join(data_dir, 'ytrain' + key + '.np'), 'wb') as f:
        np.save(f, Y_train)
    print 'X_test...'
    with open(os.path.join(data_dir, 'xtest'  + key + '.np'), 'wb') as f:
        np.save(f, X_test)
    print 'Y_test...'
    with open(os.path.join(data_dir, 'ytest'  + key + '.np'), 'wb') as f:
        np.save(f, Y_test)
    print 'Data saved.'
    return X_train, Y_train, X_test, Y_test

def build_normalized_data(imageType = 'energy', key = '_', train_split = 0.9) :
    X_train, Y_train, X_test, Y_test = build_data(imageType, 'norm_' + key, train_split)
    X_train -= 86
    X_train /= 255
    X_test  -= 86
    X_test  /= 255
    return X_train, Y_train, X_test, Y_test

def load_normalized_data(key = '_') :
    X_train, Y_train, X_test, Y_test = load_data('norm_' + key)
    X_train -= 86
    X_train /= 255
    X_test  -= 86
    X_test  /= 255
    return X_train, Y_train, X_test, Y_test

def load_data(key = '_') :
    data_dir = os.path.join(os.getcwd(), '../data')
    print 'Loading data for:'
    print 'X_train...'
    with open(os.path.join(data_dir, 'xtrain' + key + '.np'), 'rb') as f:
        X_train= np.load(f)
    print 'Y_train...'
    with open(os.path.join(data_dir, 'ytrain' + key + '.np'), 'rb') as f:
        Y_train= np.load(f)
    print 'X_test...'
    with open(os.path.join(data_dir, 'xtest'  + key + '.np'), 'rb') as f:
        X_test= np.load(f)
    print 'Y_test...'
    with open(os.path.join(data_dir, 'ytest'  + key + '.np'), 'rb') as f:
        Y_test= np.load(f)
    print 'Data loaded.'
    return X_train, Y_train, X_test, Y_test


def build_empty_model(X_shape, Y_shape) :
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape= X_shape[1:]))
    model.add(Convolution2D(32, 3, 3, activation='relu'))       
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(64, 3, 3, activation='relu'))      
    model.add(Convolution2D(64, 3, 3, activation='relu'))      
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())                                        
    model.add(Dense(128, activation='relu'))                    
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

def error_rate_for_thresholds(output, Y_test) :
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
                else:
                    cn += 1
            else:
                if output[it] >= thr:
                    fp += 1
                else:
                    fn += 1
        print 'thr: ' + str(thr),
        print 'acc: ' + str(round((cp + cn) / float(len(output)), 2)),
        print '  %fn: ' + str(round((fn) / float(len(output)), 2)),
        print '  %fp: ' + str(round((fp) / float(len(output)), 2)),
        print '  %cn: ' + str(round((cn) / float(len(output)), 2)),
        print '  %cp: ' + str(round((cp) / float(len(output)), 2))

def print_accuracy(out, test, thr = 0.5) :
    acc = 0
    pos = 0
    for it in range(len(out)) :
        if (out[it] >= thr) == test[it]:
            acc += 1
        if test[it] == 1:
            pos += 1

    acc = acc/float(len(out))
    pos = 1 - pos/float(len(out))
    print "%.4f" % round(acc,4),
    print "%.4f" % round(pos,4),

def print_error_rate_per_category(output, Y_test, thr = 0.5) :
    if output.shape[1] == 1:
        print_accuracy(output, Y_test, thr)
        print

    else :
        print '    acc    tot'
        for it in range(output.shape[1]):
            print str(it) + ':' + (' ' if it < 10 else ''),
            out_part = output[:,it]
            test_part = Y_test[:,it]
            print_accuracy(out_part, test_part, thr)
            print

def build(X_train, Y_train, X_test = None, Y_test = None, weights_filename = None, epochs = 30) :
    model = build_empty_model(X_train.shape, Y_train.shape)

    if weights_filename == None :
        if X_test == None or Y_test == None:
            print 'Hij gaat crashen: Geen validatieset als input van build()'

        model.fit(X_train, Y_train, 
            batch_size = 32, nb_epoch = epochs, 
            validation_data= (X_test, Y_test))
    else :
        model.load_weights(weights_filename)

    return model

def save_weights(model, weights_filename = 'weights') :
    filedir = os.path.join(os.getcwd())
    filename = os.path.join(filedir, 'weights_filename')
    model.save_weights(weights_filename)

if __name__ == '__main__':
    
    # Je kunt bijv build_data(imageType = 'energy', key = 'en') en load_data(key = 'en') 
    # doen om bij devolgende keer niet je data opnieuw te hoeven bouwen.
    # Als het goed is werkt dit ook voor andere imageTypes dan energy. Nog niet getest.   

    X_train, Y_train, X_test, Y_test = build_data()
    X_train, Y_train, X_test, Y_test = load_data()
    
    X_train -= 86
    X_train /= 255
    X_test  -= 86
    X_test  /= 255

    #Als de laatste param iets anders is dan 'none' gebruikt hij dat weightsbestand
    model = build(X_train, X_test, weights_filename = 'none')
    
    output = model.predict(X_test)
    print_error_rate_per_category(output, Y_test)

    filedir = os.path.join(os.getcwd())
    filename = os.path.join(filedir, 'weights_cats')

    # Hiermee kun je weights opslaan 
    #

# Die vorige was met / 255 en -mean
# met /255 en -86, incl same_number_of_idxs voor train & test, batch_size = 16, nb_epoch = 50: rond de 75%
# met /255 en -86, incl same_number_of_idxs voor train & test, batch_size = 64, nb_epoch = 50: rond de 76%
# met /255 en -86, incl same_number_of_idxs voor train & test, batch_size = 64, nb_epoch =100: wss moet je stoppen rond de epoch 75. Hij zit rond de 77%


# Netwerken trainen deligeren

# Bayes op prepDir testen.
# Harmonicity
# Voorspelbaarheid geluid