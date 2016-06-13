import sys, os
import yaml, h5py
import numpy as np
### from preprocessing.runPrep import prepFile
import pycpsp.files as files
import matplotlib.pyplot as plt
import feat_extraction.

label_dir = os.path.join(os.getcwd(), 'labeling')
sys.path.insert(0, label_dir)
import CNN.cnn as cnn

keys = ['energy', 'morphology', 'tau 1.0', 'tau 2.0', 'tau 4.0']

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

label_dir = os.path.join(os.getcwd(), 'labeling')
sys.path.insert(0, label_dir)
import CNN.cnn as cnn

def makeStartEndPairs(winsize, winstride, signals):
    # Check where the zero/infinity columns to only classify the real parts
    zeroColLeft = 0
    zeroColRight = -1
    while max(np.transpose(signals['energy'])[zeroColLeft]) <= 0:
        zeroColLeft += 1
    while max(np.transpose(signals['energy'])[zeroColRight]) <= 0:
        zeroColRight -= 1
    
    pairs = []
    index = zeroColLeft
    while index + winsize < signals['energy'].shape[1] + zeroColRight:
        pairs.append([index, index + winsize])
        index += winstride
    return pairs

def makeWindowDic(startEndPairs, signals, winsize):
    windows = {}
    for key in keys:
        if key not in signals:
            continue
        tempWindows = []
        for win in startEndPairs:
            window = [signals[key][:, win[0]:win[1]], 1, 109, winsize]
            tempWindows.append(window)
        windows[key] = tempWindows
    return windowDic
        
def classifyWindows(CNNdir, categories, startEndPairs, signals, windowDic):
    predictions = {}
    
    for category in labelDictkeys():
        tmpPredictions = []
        for key in keys:
            if key not in signals:
                continue
            
            ## Run the relevant neural net, if it exists
            weights_filename = os.path.join(weightsPath, str(category) + '_' + key + '.cnn')
            if os.path.isfile(weights_filename):
                model = cnn.build_empty_model([1, 1, 109, winsize], [1, 1])
                model.load_weights(weights_filename)
                tmp1 = model.predict(windowDic[key])
            
            ## Run the other classification methods
            # classifier = 
            
            
            ## Now combine the classifications
            for index, window in enumerate(startEndPairs):
                
        
        predictions[category] = tmpPredictions
                
    
def classifyFile(categories, CNNdir, soundFile, windowFile):
    print 'Classifying:', soundFile 
    winsize = windowFile['windows'][0]['size']
    winstride = windowFile['windows'][0]['stride']
    
    # Preprocess the file
    ### prepFile(soundFile)

    # Read the file
    filepointer = h5py.File(soundFile, 'r')
    signals = files.signalsFromHDF5(soundFile)
    
    startEndPairs = makeStartEndPairs(winsize, winstride, signals)
    windowDic = makeWindowDic(startEndPairs, signals, winsize)
    windowPredictions = classifyWindows(CNNdir, categories, startEndPairs, windowDic)
    
    return windowPredictions, filePrediction