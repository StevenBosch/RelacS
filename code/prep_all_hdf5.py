import sys, pickle
import numpy as np
from preprocessing.runPrep import prepDir
# import CNN.cnn as cnn
import labeling.windows_to_images

def build_data(hdf5Dir, labelDir, imageType) :
    windows = pickle.load(open(os.path.join(labelDir, 'labels.pickle'), 'rb'))
    X_train, Y_train, X_test, Y_test = windows_to_images.to_image_data_file_split(windows, 0.75, hdf5Dir, imageType)

    xshape = (X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    yshape = (X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    X_train = np.reshape(X_train, xshape)
    X_test = np.reshape(X_test, yshape)

    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    '''
    if len(sys.argv) != 2 or sys.argv[1] not in ['energy', 'morphology', 'tau 1.0', 'tau 2.0', 'tau 4.0']:
        print "Usage: python pipe.py imageType"
        print "imagetype should be in ['energy', 'morphology', 'tau 1.0', 'tau 2.0', 'tau 4.0']"
        sys.exit(1)
    '''

    hdf5Dir = '../sound_files/hdf5/'
    # labelDir = 'labeling/'
    # imageType = 'original'

    # Preprocess the images
    prepDir(hdf5Dir)

    # Build the data for the cnn
    # The third argument is the type of images you want to use for the cnn
    # Options are: 'original', 'morphology', 'tau 1.0', 'tau 2.0', 'tau 4.0'
    # X_train, Y_train, X_test, Y_test = build_data(hdf5Dir, labelDir, imageType)

    # Build and fit the cnn
    # network = cnn.build(X_train, Y_train, X_test, Y_test)

    # Test the network
    # accuracy = cnn.test(network, X_test, Y_test)
