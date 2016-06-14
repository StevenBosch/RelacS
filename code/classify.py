import sys, os
import yaml, h5py
import numpy as np
from preprocessing.runPrep import prepFile
import pycpsp.files as files
import matplotlib.pyplot as plt
import feat_extraction.freq_int_hist as fih

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
    windowDic = {}
    for key in keys:
        if key not in signals:
            continue
        tempWindows = []
        for win in startEndPairs:
            window = signals[key][:, win[0]:win[1]]
            window = np.reshape(window, (1, 109, winsize))
            tempWindows.append(window)
        windowDic[key] = tempWindows
    return windowDic
        
def classifyWindows(dirs, startEndPairs, signals, windowDic, winsize):
    predictions = {}
    
    for key in keys:
        print "processing", key, "files"
        if key not in signals:
            continue
        
        ### Run the other classification methods
        classifier2 = False
        tmp2 = []
        classifier_filename = os.path.join(dirs['fihs'], key + '.pickle')
        if os.path.isfile(classifier_filename):
            classifier = fih.fih(file = classifier_filename)
            classifier2 = True
    
        for category in labelDict.keys():
            tmp = np.zeros(np.asarray(startEndPairs).shape[0])
            tmp1 = []
            if classifier2:
                tmp2 = classifier.predict(np.asarray(windowDic[key]))[:, category]
            network = False
            
            ## Run the relevant neural net, if it exists
            weights_filename = os.path.join(dirs['networks'], str(category) + '_' + key + '.cnn')
            if os.path.isfile(weights_filename):
                #print windowDic[key]
                model = cnn.build_empty_model([1, 1, 109, winsize], [1, 1])
                model.load_weights(weights_filename)
                tmp1 = model.predict(np.asarray(windowDic[key]))[:,0]
                network = True
            
            ### Now combine the classifications
            if network and classifier2:
                tmp = [(x + y)/2 for x, y in zip(tmp1, tmp2)]
            elif network:
                tmp = tmp1
            elif classifier2:
                tmp = tmp2
            
            if labelDict[category] in predictions:
                predictions[labelDict[category]] = [(x + y)/2 for x, y in zip(tmp, predictions[labelDict[category]])]
            else:
                predictions[labelDict[category]] = tmp

    return predictions
    
def classifyFile(dirs, soundFile, windowFile):
    print '### Classifying:', soundFile 
    winsize = windowFile['windows'][0]['size']
    winstride = windowFile['windows'][0]['stride']
    
    # Preprocess the file
    prepFile(soundFile)

    # Read the file
    filepointer = h5py.File(soundFile, 'r')
    signals = files.signalsFromHDF5(soundFile)
    
    startEndPairs = makeStartEndPairs(winsize, winstride, signals)
    windowDic = makeWindowDic(startEndPairs, signals, winsize)
    windowPredictions = classifyWindows(dirs, startEndPairs, signals, windowDic, winsize)
    
    filePredictions = {}
    for key in windowPredictions.keys():
        filePredictions[key] = sum(windowPredictions[key])/len(windowPredictions[key])
    
    windowPredictions['windows'] = startEndPairs
    print filePredictions
    
    return windowPredictions, filePredictions