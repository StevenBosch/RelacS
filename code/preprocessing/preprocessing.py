import sys
import cv2, h5py, numpy as np
import os, matplotlib.pyplot as plt

# Change search directory for modules to include pycpsp in parent
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import pycpsp.plot as plot
import pycpsp.files as files
import pycpsp.bgmodel.config as bgconfig
import pycpsp.bgmodel.bgmodel as bgmodel

def erode(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(image, kernel, iterations = 5)

def dilate(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(image, kernel, iterations = 3)

def savePlot(image, title, filename, metadata, run = False):
    plot.plot2D(title, image, metadata=metadata)
    if run is True:
        plt.savefig('../../sound_files/preprocessed/' + filename + '/'+ title + '.png')
    else:
        plt.savefig('tmp/' + title + '.png')
    plt.close()

def foregroundBackground(refsignal, filename, metadata, run = False):
    definitions = [bgconfig.getDefaults(tau) for tau in bgconfig.tau(0.5, 4)]
    bgmodels = [bgmodel.BGModel(d['tau'], d,) for d in definitions]
    # calculate a very fast responding background model to smooth away residual noise
    refbg = bgmodel.BGModel('tau 0.1', bgconfig.getDefaults(0.1),).calculate(refsignal)
    for model in bgmodels:
        response = model.calculate(refsignal)

        savePlot(response, 'tau {}'.format(model.name), filename, metadata, run)
        # plot.plot1D('frequency-averaged energy', np.sum(refsignal, axis=0) / refsignal.shape[0], ylim=[0,60])
        # plot.plot1D('frequency-averaged bg model tau:{}'.format(model.name), np.sum(response, axis=0) / refsignal.shape[0], ylim=[0,60])
        savePlot(refbg - response, 'tau 0.1 minus tau {}'.format(model.name), filename, metadata, run)#, mask=[0,20], scale=None)

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("usage:", sys.argv[0], "<name.hdf5>")
        sys.exit(0)

    filename = sys.argv[1]
    filepointer = h5py.File(filename, 'r')
    signals = files.signalsFromHDF5(filename)
    metadata = files.metadataFromHDF5(filename)

    # Original image
    original = signals['energy'][:, metadata.get('starttime 0'):]
    refsignal = plot.imgmask(plot.imscale(signals['energy'],[0,60]), [20,80])

    # preprocess images
    eroded = erode(refsignal)
    dilated = dilate(refsignal)
    combined = dilate(erode(refsignal))
    foregroundBackground(refsignal, f)

    # Save the plots
    savePlot(refsignal, 'refsignal', metadata)
    savePlot(original, 'original', metadata)
    savePlot(eroded, 'eroded', metadata)
    savePlot(dilated, 'dilated', metadata)
    savePlot(combined, 'combined', metadata)
