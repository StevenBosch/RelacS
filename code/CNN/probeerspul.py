
import sys
import os

import numpy as np
import cnn

import matplotlib.pyplot as plt

def plot_intensities_image(cnn_input, pos_idcs):
    neg = np.copy(cnn_input)
    pos = np.copy(cnn_input)
    for idx in range(cnn_input.shape[0]):
        if pos_idcs[idx] :
            neg[idx] = 0
        else :
            pos[idx] = 0

    neg = np.sum(neg, axis = 1) #Reshaping
    pos = np.sum(pos, axis = 1) #Reshaping

    print (len(pos_idcs) - sum(pos_idcs), sum(pos_idcs), len(pos_idcs)) 
    neg2 = np.apply_along_axis(lambda x: np.histogram(x, 256, range = (1,256)), 0, neg)
    pos2 = np.apply_along_axis(lambda x: np.histogram(x, 256, range = (1,256)), 0, pos)
    
    pos = pos2[0]
    neg = neg2[0]

    pos = np.array([np.array(x, dtype = float) for x in pos])
    neg = np.array([np.array(x, dtype = float) for x in neg])

    # neg = np.sum(neg , axis= 0)
    # pos = np.sum(pos , axis= 0)

    pos /= sum(pos_idcs)# * cnn_input.shape[0]
    neg /= (len(pos_idcs) - sum(pos_idcs))# * cnn_input.shape[0]
    plot_images(pos, neg)

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

def plot_images(pos, neg) :    
    dif = pos - neg
    plt.figure(1)

    plt1 = plt.subplot(311)
    plt.imshow(pos)
    plt.colorbar()

    plt2 = plt.subplot(312)
    plt.imshow(neg)
    plt.colorbar()

    plt3 = plt.subplot(313)
    plt.imshow(dif)
    plt.colorbar()

    plt.show()



X_train, Y_train, X_test, Y_test = cnn.load_data()

# X_train -= 86
# X_train /= 255
# X_test  -= 86
# X_test  /= 255

# plt.hist(X_train.flatten(), bins = 200)
# plt.show()

# plt.hist(X_test.flatten(), bins = 200)
# plt.show()
X_train = np.sum(X_train, axis = 3)
X_train = X_train/128
print 'compressed to vertical stripes'

# plot_intensities_pixel(X_train,  X_train > 86)

print 'stressful'
plot_intensities_image(X_train,  Y_train[:,0])
print 'relaxing'
plot_intensities_image(X_train,  Y_train[:,1])
print 'sudden'
plot_intensities_image(X_train,  Y_train[:,2])
print 'Human'
plot_intensities_image(X_train,  Y_train[:,3])
print 'Traffic'
plot_intensities_image(X_train,  Y_train[:,4])
print 'Noise'
plot_intensities_image(X_train,  Y_train[:,5])
print 'Mechanical'
plot_intensities_image(X_train,  Y_train[:,6])
print 'Silence'
plot_intensities_image(X_train,  Y_train[:,7])
print 'Nature'
plot_intensities_image(X_train,  Y_train[:,8])
print 'Music'
plot_intensities_image(X_train,  Y_train[:,9])
print 'Machine'
plot_intensities_image(X_train,  Y_train[:,10])    



# stress = np.copy(X_train)
# stress[X_test == 0.] = 0
# stress = np.sum(stress, axis = (0,1,3))

# no_stress = np.copy(X_train)
# no_stress[X_test == 1.] = 0
# no_stress = np.sum(no_stress, axis = (0,1,3))




