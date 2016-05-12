import windows_to_images

import matplotlib.pyplot as plt

import os
import pickle
import h5py

import numpy as np
import theano

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD



#load the windows from file
windows = pickle.load(open('labels.pickle', 'rb'))

#cut out the actual cochleogram fragments 
# and generate a list of corresponding flags (stressfull, relaxing, sudden)
X_train, Y_train, X_test, Y_test = windows_to_images.toImageData(windows, 0.75)

print X_train.shape
print X_test.shape
print Y_train.shape
print Y_test.shape

xshape = (X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
yshape = (X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
X_train = np.reshape(X_train, xshape)
X_test = np.reshape(X_test, yshape)


model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(32, 3, 3, border_mode ='valid', input_shape= X_train.shape[1:]))
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

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(Y_train.shape[1]))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer= sgd,
    loss='binary_crossentropy',
    metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size = 16, nb_epoch=50)
score = model.evaluate(X_test, Y_test, batch_size = 16)

filedir = os.path.join(os.getcwd())
filename = os.path.join(filedir, 'testWeights')
model.save_weights(filename)