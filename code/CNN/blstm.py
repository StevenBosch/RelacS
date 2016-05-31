
import cnn
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Merge
from keras.optimizers import SGD


def build_empty_model(X_shape, Y_shape) :
    model = Sequential()
    left = Sequential()
    left.add(LSTM(output_dim=X_shape[1], return_sequences=True,
                  input_shape=X_shape[1:]))
    right = Sequential()
    right.add(LSTM(output_dim=X_shape[1], return_sequences=True,
                   input_shape=X_shape[1:], go_backwards=True))

    model.add(Merge([left, right], mode='concat'))

    model.add(Dropout(0.2))

    model.add(LSTM(512, return_sequences=False))

    if len(Y_shape) == 1:
        Y_shape = np.array([Y_shape, 1])
    model.add(Dense(Y_shape[1]))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = cnn.load_data()
    Y_train = Y_train[:, 0]
    Y_test = Y_test[:, 0]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[2], X_train.shape[3]))
    X_train = X_train.transpose((0, 2, 1))

    X_test = X_test.reshape((X_test.shape[0], X_test.shape[2], X_test.shape[3]))
    X_test = X_test.transpose((0, 2, 1))

    print X_train.shape


    X_train -= 86
    X_train /= 255
    X_test -= 86
    X_test /= 255

    X_train, Y_train = cnn.same_number_of_idxs(X_train, Y_train)
    X_test, Y_test = cnn.same_number_of_idxs(X_test, Y_test)

    print Y_train.shape
    model = build_empty_model(X_train.shape, Y_train.shape)

    model.fit([X_train, X_train], Y_train, batch_size=64, nb_epoch=100, validation_data=([X_test, X_test], Y_test))


# With the output_dim of the right lstm at 512: test accuracy fluctuated between 80 and 72%.
# 36 epochs: test accuracy fluctuated between 81 and 70%. Mostly between 74 and 77.5 %
