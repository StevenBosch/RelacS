import h5py
import os

import numpy as np


def toImageData(wins):

    totallen = 0
    for f in wins:
        totallen  = totallen  + len(wins[f])

    sample = wins.values()[0][0]
    winlen = sample['end'] - sample['start']

    frags = np.empty([totallen, 109, winlen])
    flags = np.empty([totallen, 3])

    filedir = os.path.join(os.getcwd(), 'hdf5')

    counter = 0
    skipcount = 0

    for f in wins:
        filename = os.path.join(filedir, f)
        filepointer = h5py.File(filename, 'r')
            
        for window in wins[f]:

            # add cochleogram image window to input
            beg = int(window['start'])
            end = int(window['end'])
            
            scale = filepointer.attrs.get('scale', [0, 60])

            fragment = filepointer['energy'][:, beg:end]
            if frags[0,:,:].shape != fragment.shape :
                skipcount += 1
                #Skip fragments with unexpected shape ( not frags[0,:,:].shape ) 
                continue;
            
            frags[counter, :, :] = fragment

            # add corresponding output flags to output flags array
            flagment = np.array([1.0 if window['stressful'] else 0.0, 
                1.0 if window['relaxing'] else 0.0,
                1.0 if window['sudden'] else 0.0])

            flags[counter, :] = flagment

            counter += 1
            totcount = counter + skipcount
            progress = (totcount / float(totallen))
            loadbar = '#' * int(round(20*progress)) +  ' ' * int(round(20*(1-progress)))

            print '\r[{0}] {1}% of {2} windows processed, {3} windows skipped'.format(loadbar, 
                int(round(100*progress)),
                totallen,
                skipcount),
    print #this is a newline
    return frags[:totallen-skipcount:,:], flags[:totallen-skipcount,:]

#fig, ax = plt.subplots(figsize=(20,4))
#image = ax.imshow(filepointer['energy'][:,beg:end], aspect='auto', interpolation='none', origin='bottom')
#fig.colorbar(image)
#plt.show()
            
