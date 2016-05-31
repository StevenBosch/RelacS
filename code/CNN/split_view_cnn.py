import cnn
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Merge
from keras.optimizers import SGD

def conv_part(X_shape, Y_shape):
    mod = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.

    mod.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=X_shape[1:]))
    mod.add(Activation('relu'))
    mod.add(Convolution2D(32, 3, 3))
    mod.add(Activation('relu'))
    mod.add(MaxPooling2D(pool_size=(2, 2)))
    mod.add(Dropout(0.25))

    mod.add(Convolution2D(64, 3, 3, border_mode='valid'))
    mod.add(Activation('relu'))
    mod.add(Convolution2D(64, 3, 3))
    mod.add(Activation('relu'))
    mod.add(MaxPooling2D(pool_size=(2, 2)))
    mod.add(Dropout(0.25))

    mod.add(Flatten())
    return mod

def build_empty_model(up_shape, middle_shape, down_shape, Y_shape) :
    model = Sequential()

    up = conv_part(up_shape, Y_shape)
    middle = conv_part(middle_shape, Y_shape)
    down = conv_part(down_shape, Y_shape)

    model.add(Merge([up, middle, down], mode='concat'))

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
    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def split(X) :
    third = X.shape[2]/3
    up = X[:,:, :third + 5, :]
    down = X[:,:, third*2 - 5:, :]
    middle = X[:,:, third - 5: 2*third + 5, :]
    return up, middle, down

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = cnn.load_data()
    Y_train = Y_train[:, 0]
    Y_test = Y_test[:, 0]

    X_train -= 86
    X_train /= 255
    X_test -= 86
    X_test /= 255

    X_train, Y_train = cnn.same_number_of_idxs(X_train, Y_train)
    X_test, Y_test = cnn.same_number_of_idxs(X_test, Y_test)

    up_train, middle_train, down_train = split(X_train)
    up_test, middle_test, down_test = split(X_test)

    model = build_empty_model(up_train.shape, middle_train.shape, down_train.shape, Y_train.shape)

    model.fit([up_train, middle_train, down_train], Y_train,
              batch_size=64, nb_epoch=100,
              validation_data=([up_test, middle_test, down_test], Y_test))