
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


def compute_ccls(cnn_input, pos_idcs, bins):
    pos = np.copy(cnn_input)
    neg = np.copy(cnn_input)
    for idx in range(cnn_input.shape[0]):
        if pos_idcs[idx] :
            neg[idx] = 0
        else :
            pos[idx] = 0

    pos = np.sum(pos, axis = 1) #Reshaping
    neg = np.sum(neg, axis = 1) #Reshaping

    pos = np.apply_along_axis(lambda x: np.histogram(x, bins, range = (0,256))[0], 0, pos)
    neg = np.apply_along_axis(lambda x: np.histogram(x, bins, range = (0,256))[0], 0, neg)
    
    pos = np.array([np.array(x, dtype = float) for x in pos])
    neg = np.array([np.array(x, dtype = float) for x in neg])

    tot_pos_per_freq = np.sum(pos, axis = 1)
    tot_neg_per_freq = np.sum(pos, axis = 1)

    ccl_pos = np.empty(pos.shape)
    ccl_neg = np.empty(neg.shape)
    for it in range(len(tot_pos_per_freq)) :
        ccl_pos[it] = pos[it] / float(tot_pos_per_freq[it])
        ccl_neg[it] = neg[it] / float(tot_neg_per_freq[it])


    ccl_pos[ccl_pos == 0] = 0.0000001
    ccl_neg[ccl_neg == 0] = 0.0000001

    pos_frac = (sum(pos_idcs)/len(pos_idcs))
    print pos_frac
    pos /= pos_frac
    neg /= 1-pos_frac 
    plot_images(pos, neg, pos-neg)

    return ccl_pos.transpose(1,0), ccl_neg.transpose(1,0)

def plot_intensities_pixel(cnn_input, pos_idcs):
    neg = np.copy(cnn_input)
    neg[pos_idcs == True] = 0
    neg[pos_idcs == False] = 1
    neg = np.sum(neg , axis= (0,1))

    pos = np.copy(cnn_input)
    pos[pos_idcs == False] = 0
    pos[pos_idcs == True] = 1
    pos = np.sum(pos , axis=(0,1))

    plot_intensities(pos, neg)

def plot_intensities(pos, neg) :
    plt.plot(pos)
    plt.plot(neg)
    plt.show()

def plot_images(pos, neg, dif) :    
    plt.figure(1)

    plt1 = plt.subplot(311)
    plt.imshow(pos[1:].transpose(1,0))
    plt.colorbar()

    plt2 = plt.subplot(312)
    plt.imshow(neg[1:].transpose(1,0))
    plt.colorbar()

    plt3 = plt.subplot(313)
    plt.imshow(dif[1:].transpose(1,0))
    plt.colorbar()

    plt.show()

def compute_ccls_per_category(bins, X_train, Y_train) :
    print 'computing class conditional log likelihoods...'
    X_train = np.sum(X_train, axis = 3)
    X_train = X_train/128
  
    ccls_pos = np.empty((Y_train.shape[1], 109, bins))
    ccls_neg = np.empty((Y_train.shape[1], 109, bins))

    cats = ['stressful', 'relaxing', 'sudden', 'Human', 'Traffic', 'Noise', 'Mechanical', 'Silence', 'Nature', 'Music', 'Machine']

    for idx in range(Y_train.shape[1]):
        print cats[idx]
        ccls_pos[idx], ccls_neg[idx] = compute_ccls(X_train,  Y_train[:,idx], bins) 

    return ccls_pos, ccls_neg

def compute_probabilities(bins, X_test, Y_test, ccls_pos, ccls_neg, a_priory) :
    X_test = np.sum(X_test, axis = (1,3))
    X_test = X_test/128

    prob_pos = np.zeros((X_test.shape[0], Y_test.shape[1]))
    prob_neg = np.zeros((X_test.shape[0], Y_test.shape[1]))
    for idx in range(X_test.shape[0]):
        print '\rcomputing actual probabilities... ' + str(idx+1) + '/' + str(X_test.shape[0]),
        for cat in range(Y_test.shape[1]) :
            prob_pos[idx, cat] = np.log(a_priory[cat])
            prob_neg[idx, cat] = np.log(1-a_priory[cat])
            for freq in range(X_test.shape[1]) :
                intensity_idx = int( X_test[idx, freq] * bins/256.0)
                prob_pos[idx, cat] += np.log(ccls_pos[cat, freq, intensity_idx])
                prob_neg[idx, cat] += np.log(ccls_neg[cat, freq, intensity_idx])
    print
    return prob_pos, prob_neg

def bayes(X_train, Y_train, X_test, Y_test, pyramid_height = 1, max_bins = 256, show_train_acc = True) :
    print 'computing a priory log likelihoods...'
    a_priory = np.sum(Y_train, axis = 0)/ Y_train.shape[0]


    output = np.zeros((pyramid_height , Y_test.shape[0], Y_test.shape[1]))

    for n in range(pyramid_height ) :
        bins = max_bins/ 2**n
        print 'binsize: ' + str(bins)
        
        ccls_pos, ccls_neg = compute_ccls_per_category(bins, X_train, Y_train)
        
        print 'Test:'
        prob_pos, prob_neg = compute_probabilities(bins, X_test, Y_test, ccls_pos, ccls_neg, a_priory)
        output[n] = prob_pos > prob_neg
        cnn.print_error_rate_per_category(output[n], Y_test, thr = 0.5)

        if show_train_acc :
            print 'Train:'
            prob_pos, prob_neg = compute_probabilities(bins, X_train, Y_train, ccls_pos, ccls_neg, a_priory)
            train_output = prob_pos > prob_neg
            cnn.print_error_rate_per_category(train_output, Y_train, thr = 0.5)
            print

    if pyramid_height != 1 :
        print 'Performance with all bin sizes together:'
        compressed_output = np.sum(output, axis = 0)
        compressed_output = compressed_output/pyramid_height 
        cnn.print_error_rate_per_category(compressed_output, Y_test, thr = 0.5)

def build_empty_model(X_shape, Y_shape) :
    model = Sequential()

    model.add(Dense(1024, input_shape= X_shape[1:]))
    model.add(Activation('relu'))

    if len(Y_shape) == 1:
        Y_shape = np.array([Y_shape, 1])
    model.add(Dense(Y_shape[1]))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer= sgd,
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model


def mlp(X_train, Y_train, X_test, Y_test) :
    X_train = np.sum(X_train, axis = (1,3))
    X_train = X_train/128
    X_train -= 65
    X_train /= 256

    X_test = np.sum(X_test, axis = (1,3))
    X_test = X_test/128
    X_test -= 65
    X_test /= 256

    print 'Building model...'
    model = build_empty_model(X_train.shape, Y_train.shape)
    
    model.fit(X_train, Y_train, batch_size = 32, nb_epoch = 200, validation_data= (X_test, Y_test))

    print 'predicting output'
    output = model.predict(X_test)

    cnn.print_error_rate_per_category(output, Y_test)

def naive(X_train, Y_train, X_test, Y_test) :
    X_train = np.sum(X_train, axis = (1,3))
    X_train = np.mean(X_train, axis = 0)
    print 'Not done yet'

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = cnn.load_data()
    
    bayes(X_train, Y_train, X_test, Y_test)

    # mlp(X_train, Y_train[:,0], X_test, Y_test[:,0])

    # naive(X_train, Y_train[:,0], X_test, Y_test[:,0])

    # X_train -= 86
    # X_train /= 255
    # X_test  -= 86
    # X_test  /= 255

    # plt.hist(X_train.flatten(), bins = 200)
    # plt.show()

    # plt.hist(X_test.flatten(), bins = 200)
    # plt.show()

    # plot_intensities_pixel(X_train,  X_train > 86)


'''
BAYES OUTPUT: 

Loading data from:
X_train...
Y_train...
X_test...
Y_test...
Data loaded.
computing a priory log likelihoods...
binsize: 256
computing class conditional log likelihoods...
Test:
computing actual probabilities... 1345/1345 
    acc    tot
0:  0.8119 0.7747
1:  0.8119 0.8119
2:  0.7435 0.7435
3:  0.8245 0.8245
4:  0.8766 0.8766
5:  0.8743 0.8743
6:  0.9941 0.9941
7:  1.0000 1.0000
8:  0.9509 0.9509
9:  0.9375 0.9375
10: 0.9814 0.9814

Train:
computing actual probabilities... 12009/12009 
    acc    tot
0:  0.7155 0.6244
1:  0.8943 0.8943
2:  0.7995 0.7903
3:  0.7389 0.7382
4:  0.8301 0.8292
5:  0.9046 0.9046
6:  0.9714 0.9714
7:  0.9800 0.9800
8:  0.9646 0.9646
9:  0.9122 0.9109
10: 0.9745 0.9744
'''
