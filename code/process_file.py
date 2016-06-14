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
    for key in predictions.keys():
        if not key == 'windows':
            plt.plot(predictions[key])
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
    filepointer = h5py.File(soundFile, 'rw')
    signals = files.signalsFromHDF5(soundFile)    
    
    attrs = filepointer.attrs
    fs = filepointer['energy'].shape[1] / attrs['duration']
    
    windowPredictions, filePredictions = classifyFile(dirs, soundFile, windows)
    # windowPredictions = pickle.load( open( "windowPredictions.pickle", "rb" ) )
    
    # Turn frames back to time
    # for frame in windowPredictions['windows']:
                
    plotResults(windowPredictions)

    
    # Store everything to be processed by the site
    with open('windowPredictions.pickle', 'w') as f :
        pickle.dump(windowPredictions, f)

    #with open('filePredictions.pickle', 'w') as f :
    #    pickle.dump(filePredictions, f)