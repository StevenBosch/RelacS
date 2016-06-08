import sys, yaml
import numpy as np
from preprocessing.runPrep import prepFile
import CNN.cnn

def classifyWindow(neuralNetPath, signals, winsize, winstride):
    keys = ['energy', 'morphology', 'tau 1.0', 'tau 2.0', 'tau 4.0']

    prediction = 0
    for key in keys:
        window = signals[key][index:index + winsize]
        # Get CNN prediction
        model = cnn.build(X_train, Y_train, weights_filename = neuralNetPath + key + '.cnn')
        cnnPredict = model.predict(window)
        
        # Get Bayes prediction
        
        # Get other method's predictions
        
        
        # prediction += (cnnPredict + rogierPredcit + pimPredict)/3
    return = prediction / len(keys) # Or some voting?

def classifyFile(neuralNetPath, soundFile, winsize, winstride):
    # Preprocess the file
    prepFile(soundFile)

    # Read the file
    filepointer = h5py.File(soundFile, 'r')
    signals = files.signalsFromHDF5(soundFile)

    # Check where the zero/infinity columns to only classify the real parts
    zeroColLeft = 0
    zeroColRight = -1
    while max(np.tranpose(signals['energy'])[0]) <= 0:
        zeroColLeft += 1
    while max(np.tranpose(signals['energy'])[-1]) <= 0:
        zeroColRight -= 1
    
    windowPredictions = []
    # Classify every type of preprocessed image for every slice in the file
    index = zeroColLeft
    while index + winsize < len(signals['energy']) + zeroColRight + 1:
        prediction = classifyWindow(neuralNetPath, signals, winsize, winstride)
        windowPredictions.append(prediction)
        
        # Next window
        index += winstride
        
    # Now take the averag to make the classifications of the whole file
    filePrediction = sum(windowPredictions) / len(windowPredictions)
    
    return windowPredictions, filePrediction

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: python classify.py <filename.hdf5> <windows.yaml>"
    with open(sys.argv[2], 'r') as f:
        windows = yaml.load(f)
        print windows['windows'][0]

    windowPredictions, filePrediction = classifyFile(sys.argv[1], windows['windows'][0]['size'], windows['windows'][0]['stride'])
    
    print "Window predictions =", windowPredictions
    print "File predictions =", filePredictions
