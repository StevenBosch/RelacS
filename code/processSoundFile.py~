import sys
import classify.classifyFile

def makeWavelet(signals):
    energy = signals['energy'][:,:]
    hist = np.zeros(len(energy[0]))
    for row in energy:
        hist = [x + y for x, y in zip(energy, row)]
    # Smoothing
    # hist = savgol_filter(hist, 15, 3)
    return hist

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: python processSoundFile.py <filename.hdf5> <windows.yaml>"
    with open(sys.argv[2], 'r') as f:
        windows = yaml.load(f)
        print windows['windows'][0]
    
    soundFile = sys.argv[1]
    filepointer = h5py.File(soundFile, 'r')
    signals = files.signalsFromHDF5(soundFile)    

    windowPredictions, filePrediction = classifyFile(soundFile, windows['windows'][0]['size'], windows['windows'][0]['stride'])
    wavelet = makeWavelet(signals)
    
    # Store everything to be processed by the site
    
