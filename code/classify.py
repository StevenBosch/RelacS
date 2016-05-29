import sys, yaml
import numpy as np
from preprocessing.runPrep import prepFile

def classify(soundFile, winsize, winstride): # And the needed trained neural nets?
    # Preprocess the file
    prepFile(soundFile)

    # Read the file
    filepointer = h5py.File(filename, 'r')
    signals = files.signalsFromHDF5(filename)

    # Check where the zero/infinity columns end
    # Remember the number of columns to keep the right sync with the sound file
    zeroColLeft = 0
    zeroColRight = -1
    while max(np.tranpose(signals['energy'])[0]) <= 0:
        zeroColLeft += 1
    while max(np.tranpose(signals['energy'])[-1]) <= 0:
        zeroColRight -= 1

    # Classify every type of preprocessed image for every slice in the file
    index = zeroColLeft
    while index + winsize < len(signals['energy']) + zeroColRight + 1:
        for key in ['energy', 'morphology', 'tau 1.0', 'tau 2.0', 'tau 4.0']:
            window = signals[key][index:index + winsize]
            # classify
            # cnn.predict(trained neuralNet, window)
            # Save result of prediction in text file
        index += winstride


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: python classify.py <filename.hdf5> <windows.yaml>"
    with open(sys.argv[2], 'r') as f:
        windows = yaml.load(f)
        print windows['windows'][0]

    classify(sys.argv[1], windows['windows'][0]['size'], windows['windows'][0]['stride'])
