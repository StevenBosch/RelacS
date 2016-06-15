import sys, os, shutil
from classify import classifyFile
import yaml, h5py, pickle
import pycpsp.files as files
import matplotlib.pyplot as plt
import numpy as np

def makeWavelet(signals):
    energy = signals['energy'][:,:]
    hist = []
    for row in np.transpose(energy):
        if np.any(np.asarray(row) < 0):
            hist.append(0)
        else:
            hist.append(sum(row))
    # hist = savgol_filter(hist, 15, 3)
    return hist

def getSuddenSounds(hist, signals):
    loudnessFactors = []
    differenceFactors = []
    difference = 0
    for index, value in enumerate(hist):
        if index + 1 == len(hist):
            break
        else:
            difference += abs(value - hist[index + 1])
    avgDiff = float(difference) / float(len(hist) - 1)
    
    differenceFactors.append(1)
    for index, value in enumerate(hist):
        if index + 1 == len(hist):
            break
        else:
            if float(hist[index + 1] - value) > avgDiff:
                differenceFactors.append(float(hist[index + 1] - value) / avgDiff)
            else:
                differenceFactors.append(1)
                
    avgEnergy = sum(hist)/len(hist)
    
    for value in hist:
        loudnessFactors.append(value/avgEnergy)
    
    return differenceFactors, loudnessFactors

def loudness_feature(hist, windowPredictions, split = 0.9, std_mult = 3) :
    big_part = sorted(hist)[:int(len(hist)*split)]
    threshold = np.mean(big_part) + std_mult*np.std(big_part)
    
    for index, window in enumerate(windowPredictions['windows']):
        beg, end = windowPredictions['windows'][index]
        is_loud_list = hist[beg:end] > threshold
        avg_is_loud = np.sum(is_loud_list)/float(end-beg)
        avg_is_loud *= 2
        loud_prediction = min(avg_is_loud, 1.)
        prediction = windowPredictions['stressful'][index] 
        windowPredictions['stressful'][index] = max(loud_prediction, prediction)
    
    return windowPredictions
    
def replace_last_two(source_string, replace_what, replace_with):
    first_part, sep, tail = source_string.rpartition(replace_what)
    head, sep, middle = first_part.rpartition(replace_what)

    return head + replace_with + middle + replace_with + tail

def getOriginal(predictions, soundFile):
    ID         = 0
    FILENAME   = 1
    LABELER    = 2
    START_TIME = 3
    END_TIME   = 4
    STRESSFUL  = 5
    RELAXING   = 6
    SUDDEN     = 7
    CATEGORY   = 8
    OTHER      = 9
    
    soundFile = os.path.basename(soundFile)
    for i in range(3):
        soundFile = os.path.splitext(soundFile)[0]
    
    original = {}
#    for key in predictions.keys():
#        original[key] = []
    for key in ['time', 'stressful', 'relaxing', 'sudden']:
        original[key] = []
        
    with open("labeling/labeling.csv", 'r') as f:
        soundFile2 = replace_last_two(soundFile, ':', '-')
        for line in f.readlines():
            line = line.strip()
            row = line.split(',')
            if row[FILENAME] == soundFile or row[FILENAME] == soundFile2:
                original['time'].append([float(row[START_TIME][-6:]), float(row[END_TIME][-6:])])
                original['stressful'].append(1) if row[STRESSFUL].lower() == 'yes' else original['stressful'].append(0)
                original['relaxing'].append(1) if row[RELAXING].lower() == 'yes' else original['relaxing'].append(0)
                original['sudden'].append(1) if row[SUDDEN].lower() == 'yes' else original['sudden'].append(0)
    return original
        
def plotResults(predictions, name, plotting = 'stressful'):
    time = []
    for moment in predictions['time']:
        time.append((moment[0] + moment[1])/2.0)
    
    if plotting == 'stressful':
        plt.plot(time, predictions['stressful'])
        plt.title('stressful')
        plt.savefig('tmpPlots/' + name + 'stressful' + '.png')
        plt.close()        
    else:
        for key in predictions.keys():
            if not key == 'windows':
                plt.figure(1)
                plt.plot(time, predictions[key])
                plt.title(key)
                plt.savefig('tmpPlots/' + name + key + '.png')
                plt.close()

def plotStressfullBoth(predictions, original):
    cats = ['stressful', 'relaxing', 'sudden']
    time = []
    tmp = {}
    for cat in cats:
        tmp[cat] = []
    
    for index, moment in enumerate(predictions['time']):
        mom = (moment[0] + moment[1])/2.0
        time.append(mom)
        for index2, window in enumerate(original['time']):
            if mom >= window[0] and mom <= window[1]:
                for cat in cats:
                    tmp[cat].append(original[cat][index2])
                break
            if index2 == len(original['time']) - 1:
                for cat in cats:
                    tmp[cat].append(0)
    
    for cat in cats:  
        plt.figure(1)
        plt.plot(time, predictions[cat])
        plt.plot(time, tmp[cat])
        plt.title(cat + ' both')
        plt.savefig('tmpPlots/' + cat + '_both.png')
        plt.close()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: python processSoundFile.py <filename.hdf5>"

    with open('labeling/windows.yaml', 'r') as f:
        windows = yaml.load(f)
    
    ######### DIRECTORIES #########
    dirs = {}
    dirs['networks'] = 'classifiers/trained_nets/'
    dirs['fihs'] = 'classifiers/fih'
    
    
    ######### READING FILE #########
    soundFile = sys.argv[1]
    filepointer = h5py.File(soundFile, 'r+')
    signals = files.signalsFromHDF5(soundFile)    
    
    
    ######### CLASSIFICATION  #########
    windowPredictions, filePredictions = classifyFile(dirs, soundFile, windows)
    # windowPredictions = pickle.load( open( "windowPredictions.pickle", "rb" ) )
    sudden = True
    loudness = True
    
    ######### Loudness and sudden features #########
    hist = makeWavelet(signals)
    
    if sudden:
        differenceFactors, loudnessFactors = getSuddenSounds(hist, signals)
        for index, window in enumerate(windowPredictions['windows']):
            diffFactor = float(sum(differenceFactors[window[0]:window[1]])) / float(window[1]-window[0])
            loudFactor = float(sum(loudnessFactors[window[0]:window[1]])) / float(window[1]-window[0])
            windowPredictions['stressful'][index] *= diffFactor * loudFactor
            windowPredictions['stressful'][index] = min(windowPredictions['stressful'][index], 1)
    
    if loudness:
        windowPredictions = loudness_feature(hist, windowPredictions)
    
    ######### WINDOWS TO TIMES #########
    attrs = filepointer.attrs
    fs = filepointer['energy'].shape[1] / attrs['duration']
    windowPredictions['time'] = []
    for window in windowPredictions['windows']:
        windowPredictions['time'].append([window[0]/fs, window[1]/fs])
    
    ######### PLOTTING #########
    if os.path.exists('tmpPlots/'):
        shutil.rmtree('tmpPlots')
    os.makedirs('tmpPlots/')
    plotResults(windowPredictions, 'pred_', plotting = 'stressful')
    
    # Get the labeling from labeling.yaml and plot the comparisons to our predictions
    original = getOriginal(windowPredictions, soundFile)
    if original['stressful']:
        plotResults(original, 'labeled_')
        plotStressfullBoth(windowPredictions, original)
    
    ######### STORAGE #########
    # Store everything to be processed by the site
    with open('windowPredictions.pickle', 'w') as f :
        pickle.dump(windowPredictions, f)

    with open('filePredictions.pickle', 'w') as f :
        pickle.dump(filePredictions, f)