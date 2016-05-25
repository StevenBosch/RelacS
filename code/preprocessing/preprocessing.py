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

def savePlot(image, title, filename, metadata):
    plot.plot2D(title, image, metadata=metadata)
    plt.savefig(filename + '.png')
    plt.close()

def foregroundBackground(refsignal, prepDir):
    definitions = [bgconfig.getDefaults(tau) for tau in bgconfig.tau(0.5, 4)]
    bgmodels = [bgmodel.BGModel(d['tau'], d,) for d in definitions]
    for model in bgmodels:
        response = model.calculate(refsignal)
        np.save(prepDir + 'tau {}'.format(model.name), response)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage:", sys.argv[0], "<name.hdf5>")
        sys.exit(0)

    filename = sys.argv[1]
    filepointer = h5py.File(filename, 'r')
    signals = files.signalsFromHDF5(filename)
    metadata = files.metadataFromHDF5(filename)

    # Original image
    original = signals['energy'][:,44:]

'''
    # preprocess images
    eroded = erode(original)
    dilated = dilate(original)
    combined = dilate(erode(original))
    foregroundBackground(original)

    cv2.imwrite('tmp/original.png', original)
    cv2.imwrite('tmp/eroded.png', eroded)
    cv2.imwrite('tmp/dilated.png', dilated)
    cv2.imwrite('tmp/combined.png', eroded)


    # Save the plots
    savePlot(original, 'original', 'tmp/original.png', metadata)
    savePlot(eroded, 'eroded', 'tmp/eroded.png', metadata)
    savePlot(dilated, 'dilated', 'tmp/dilated.png', metadata)
    savePlot(combined, 'combined', 'tmp/combined.png', metadata)

'''
