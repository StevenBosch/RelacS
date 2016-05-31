
import sys
import os

import numpy as np
import cnn

import matplotlib.pyplot as plt

X_train, Y_train, X_test, Y_test = cnn.load_data()
Y_train = Y_train[:, 0]
Y_test = Y_test[:, 0]

# X_train -= 86
# X_train /= 255
# X_test  -= 86
# X_test  /= 255

# plt.hist(X_train.flatten(), bins = 200)
# plt.show()

# plt.hist(X_test.flatten(), bins = 200)
# plt.show()


small = np.copy(X_train)
small[small > 86] = 0
small = np.sum(small , axis=0)
small = np.sum(small , axis=0)
print small

large = np.copy(X_train)
large[large < 86] = 0
large = np.sum(large , axis=0)
large = np.sum(large , axis=0)
print large


plt.figure(1)

plt1 = plt.subplot(211)
plt.imshow(small)
plt.colorbar()

plt2 = plt.subplot(212)
plt.imshow(large)

plt.colorbar()

plt.show()

