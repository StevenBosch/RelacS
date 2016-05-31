
import sys
import os

label_dir = os.path.join(os.getcwd(), '../labeling')
sys.path.insert(0, label_dir)

import windows_to_images
import cnn

import matplotlib.pyplot as plt

X_train, Y_train, X_test, Y_test = cnn.load_data()
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
