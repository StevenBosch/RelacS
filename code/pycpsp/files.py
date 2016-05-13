import h5py, os, glob, numpy as np

"""
    Load a signal representation from an HDF5 file
"""
def signalsFromHDF5(filepath, representations=None):
    # normalize the representations
    if type(representations) == str:
        representations = [representations]

    # check if files exist
    if not os.path.isfile(filepath):
        raise Exception("Given file {0} is not an existing file. Give a filepath relative to files.py or an absolute path".format(filepath))

    # will contain representations for each type wanted
    matrices = {}
    # total frame count
    frames = 0
    hshape = None

    # Open the file using h5py
    fh = h5py.File(filepath)

    if representations == None:
        representations = fh.keys()
    for r in representations:
        if not r in fh.keys():
            raise Exception("Given representation does not exist in HDF5 file. Available keys: {0}".format(','.join(fh.keys())))
        #read the representation and append it
        matrices[r] = fh[r]

    return matrices

"""
    Return the metadata generator object from the HDF5 file
"""
def metadataFromHDF5(filepath):
    # check if files exist
    if not os.path.isfile(filepath):
        raise Exception("Given file {0} is not an existing file. Give a filepath relative to files.py or an absolute path".format(filepath))
    fh = h5py.File(filepath)
    return fh.attrs

"""
    Downsample the signal with a given factor along the X axis
"""
def downsampleX(data, factor):
    factor = float(factor)

    # factor smaller than 1 becomes upsampling...
    if factor < 1.0:
        raise Exception("Can only downsample with factor >= 1")

    d_length = float(np.shape(data)[1])
    # simple division
    new_length = round(d_length / factor)
    # construct a lookup table with indices to pick
    lookup = np.floor(np.arange(d_length/new_length/2, d_length, factor)).astype(np.int)
    #smoothed = smoothX(data, samplerate)
    smoothed = data

    #giving an array as second dimension lets numpy pick only those indices from the array
    return smoothed[:,lookup]

"""
    Smooth the signal with a running average filter along the X axis
"""
def smoothX(data, length):
    # use a running average filter with n = length
    # take care to correct the amplitudes with 1/length
    fir = np.ones(length) * (1.0 / length)
    # concatenate all smoothed rows in vertical direction
    # this omits using conv2 (don't know which is faster,
    # but it makes the filter easier to create)
    return np.concatenate([np.array([np.convolve(row, fir, 'same')]) for row in data], axis=0)
