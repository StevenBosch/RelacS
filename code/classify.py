import sys, os
import yaml, h5py
import numpy as np
from preprocessing.runPrep import prepFile
import pycpsp.files as files
import matplotlib.pyplot as plt


label_dir = os.path.join(os.getcwd(), 'labeling')
sys.path.insert(0, label_dir)


import CNN.cnn as cnn


labelDict = { 0 : 'stressful', 
              1 : 'relaxing',
              2 : 'sudden',
              3 : 'other',
              4 : 'human',
              5 : 'traffic',
              6 : 'noise',
              7 : 'mechanical',
              8 : 'silence',
              9 : 'nature',
              10: 'music',
              11: 'machine'}

def getCNNpredictions(CNNdict, signals, beg, end) :
    prediction = {}

    window = signals[CNNdict['prepType']][:, beg:end]
    window = np.reshape(window, (1, 1, 109, 128))
    
    for label in CNNdict['nets'].keys() :
        prediction[label] = CNNdict['nets'][label].predict(window)
    
    return prediction

def classifyWindow(CNNdict, signals, beg,end):
    keys = ['energy', 'morphology', 'tau 1.0', 'tau 2.0', 'tau 4.0']

    prediction = getCNNpredictions(CNNdict, signals, beg, end)
    
    
    # Get Bayes prediction
    
    # Get other method's predictions
    
    # prediction += (cnnPredict + bayesPredict + ...)/...
    
    print
    for key in prediction:
        print key, '\t\t', prediction[key] 
    
    return  prediction

def classifyFile(CNNdict, soundFile, winsize, winstride):
    print 'Classifying:', soundFile 
    # Preprocess the file
    #### prepFile(soundFile) THIS SHOULD NOT BE COMMENTED, THIS SHOULD NOT BE COMMENTED,THIS SHOULD NOT BE COMMENTED THIS SHOULD NOT BE COMMENTED

    # Read the file
    filepointer = h5py.File(soundFile, 'r')
    signals = files.signalsFromHDF5(soundFile)

    # Check where the zero/infinity columns to only classify the real parts
    zeroColLeft = 0
    zeroColRight = -1
    while max(np.transpose(signals['energy'])[0]) <= 0:
        zeroColLeft += 1
    while max(np.transpose(signals['energy'])[-1]) <= 0:
        zeroColRight -= 1
    
    windowPredictions = []
    # Classify every type of preprocessed image for every slice in the file
    index = zeroColLeft
    while index + winsize < signals['energy'].shape[1] + zeroColRight + 1:
        prediction = classifyWindow(CNNdict, signals, index, index+winsize)

        windowPredictions.append(prediction)
        
        # Next window
        index += winstride
        
    # Now take the averag to make the classifications of the whole file
    plt.imshow(signals['energy'][:, 3000:4000])
    plt.show()
    filePrediction = sum(windowPredictions) / len(windowPredictions)
    
    return windowPredictions, filePrediction

def constructCNNDict(winsize, prepType = 'tau 1.0', weightsPath = 'CNN/trained_nets') :
    CNNdict = {}
    CNNdict['prepType'] = prepType
    CNNdict['nets'] = {}
    for key in labelDict.keys() :
        weights_filename = os.path.join(weightsPath, str(key) + '_' + prepType + '.cnn')
        
        if os.path.isfile(weights_filename) :
            model = cnn.build_empty_model([1, 1, 109, winsize], [1, 1])
            model.load_weights(weights_filename)
            CNNdict['nets'][labelDict[key]] = model
            print 'CNN for', labelDict[key], 'labeling constructed'

    return CNNdict

if __name__ == '__main__':
    # filename = '../sound_files/hdf5/C.D.A.Wieringa_2016-04-25T12:17:39.000Z.wav.2.hdf5'
    # filename = '../sound_files/hdf5/F.de.Vries.11_2016-04-15T08:25:54.000Z.wav.2.hdf5'
    filename = '../sound_files/hdf5/J.Welink_2016-04-25T11:57:21.000Z.wav.2.hdf5'

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        

    yamlfile = 'labeling/windows.yaml' 
    
    if len(sys.argv) > 2:
        yamlfile = sys.argv[2]
    
    with open(yamlfile, 'r') as f:
        windows = yaml.load(f)

    CNNdict = constructCNNDict(windows['windows'][0]['size'])
    windowPredictions, filePrediction = classifyFile(CNNdict, filename, windows['windows'][0]['size'], windows['windows'][0]['stride'])
    
    print "Window predictions =", windowPredictions
    print "File predictions =", filePredictions
