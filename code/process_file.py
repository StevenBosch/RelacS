import sys
from classify import classifyFile
import yaml, h5py, pickle
import pycpsp.files as files
import matplotlib.pyplot as plt

def makeWavelet(signals):
    energy = signals['energy'][:,:]
    hist = np.zeros(len(energy[0]))
    for row in energy:
        hist = [x + y for x, y in zip(energy, row)]
    # Smoothing
    # hist = savgol_filter(hist, 15, 3)
    return hist

#def addOriginalLabels(predictions):
#    with open("labeling/labeling.csv", 'r') as f:
        
def plotResults(predictions):
    time = []
    for moment in predictions['time']:
        time.append((moment[0] + moment[1])/2.0)
    
    for key in predictions.keys():
        if not key == 'windows':
            plt.figure(1)
            plt.plot(time, predictions[key])
            plt.title(key)
            plt.show()        

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: python processSoundFile.py <filename.hdf5> <windows.yaml>"
    with open(sys.argv[2], 'r') as f:
        windows = yaml.load(f)
    
    # Settings
    dirs = {}
    dirs['networks'] = 'CNN/trained_nets/'
    dirs['fihs'] = 'feat_extraction/classifiers/'
    
    soundFile = sys.argv[1]
    filepointer = h5py.File(soundFile, 'r')
    signals = files.signalsFromHDF5(soundFile)    
    
    # Get the window and file classifications
    windowPredictions, filePredictions = classifyFile(dirs, soundFile, windows)
    # windowPredictions = pickle.load( open( "windowPredictions.pickle", "rb" ) )
    
    # Add window times
    attrs = filepointer.attrs
    fs = filepointer['energy'].shape[1] / attrs['duration']
    windowPredictions['time'] = []
    for window in windowPredictions['windows']:
        windowPredictions['time'].append([window[0]/fs, window[1]/fs])
    
    plotResults(windowPredictions)

    # Store everything to be processed by the site
    with open('windowPredictions.pickle', 'w') as f :
        pickle.dump(windowPredictions, f)

    with open('filePredictions.pickle', 'w') as f :
        pickle.dump(filePredictions, f)