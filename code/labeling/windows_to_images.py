import h5py
import os
import random
import scipy


import numpy as np

#wins:          windows that will be used to generate images
#percentage:    percentage of files used to generate training data vs validation data
def to_image_data_file_split(wins, percentage, filedir, imageType = 'energy'):
    if percentage > 1 or percentage < 0:
        RuntimeError('Percentage should be between 0 and 1')
    split_index = int(len(wins) * percentage)
    if split_index == len(wins):
        RuntimeError('Aborting: Training partition too large. Could not construct validation/testing set')

    train_wins = dict(wins.items()[:split_index])
    valid_wins = dict(wins.items()[split_index:])

    print 'Constructing train data...'
    train_r, train_l = to_image_data(train_wins, filedir, imageType)
    print 'Constructing validation/test data...'
    valid_r, valid_l = to_image_data(valid_wins, filedir, imageType)
    return train_r, train_l, valid_r, valid_l


def to_image_data(wins, filedir, imageType):
    totallen = 0
    for f in wins:
        totallen  = totallen  + len(wins[f])

    sample = wins.values()[0][0]
    winlen = sample['end'] - sample['start']

    frags = np.empty([totallen, 109, winlen])
    flags = np.empty([totallen, 12])

    counter = 0
    skipcount = 0
    rescaled = False

    for f in wins:
        filename = os.path.join(filedir, f)
        filepointer = h5py.File(filename, 'r')

        for window in wins[f]:

            # create image from window
            beg = int(window['start'])
            end = int(window['end'])

            scale = filepointer.attrs.get('scale', [0, 60])

            fragment = filepointer[imageType][:, beg:end]

            if not rescaled and frags[0,:,:].shape[0] == fragment.shape[0] and frags[0,:,:].shape[1] == fragment.shape[1]*2 :
                rescaled = True
                temp_frags = frags
                frags = np.empty([totallen, 109, winlen/2])
                for it in range(counter) :
                    frags[it,:,:] =  scipy.misc.imresize(temp_frags[it,:,:], frags[it,:,:].shape)
                      
            if frags[0,:,:].shape[0] == fragment.shape[0] and frags[0,:,:].shape[1]*2 == fragment.shape[1] :
                fragment = scipy.misc.imresize(fragment, (fragment.shape[0], frags[0,:,:].shape[1]))
                
            if frags[0,:,:].shape != fragment.shape or not np.all(np.isfinite(fragment)):
                skipcount += 1

                #Skip fragments with unexpected shape ( not frags[0,:,:].shape )
                continue;

            # create array of corresponding output flags
            flagment = np.array([1.0 if window['stressful'] else 0.0,
                1.0 if window['relaxing'] else 0.0,
                1.0 if window['sudden'] else 0.0,
                1.0 if ('Other' in window['category']) else 0.0,
                1.0 if ('Human' in window['category']) else 0.0,
                1.0 if ('Traffic' in window['category']) else 0.0,
                1.0 if ('Noise' in window['category']) else 0.0,
                1.0 if ('Mechanical' in window['category']) else 0.0,
                1.0 if ('Silence' in window['category']) else 0.0,
                1.0 if ('Nature' in window['category']) else 0.0,
                1.0 if ('Music' in window['category']) else 0.0,
                1.0 if ('Machine' in window['category']) else 0.0])

            # add to train or validation set
            frags[counter, :, :] = fragment
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
    r = frags[:totallen-skipcount, :]
    l = flags[:totallen-skipcount, :]
    return r, l

#Outdated. Can be removed.
def to_image_data_window_split(wins, percentage, filedir, imageType = 'energy'):
    if percentage > 1 or percentage < 0:
        RuntimeError('Percentage should be between 0 and 1')

    totallen = 0
    for f in wins:
        totallen  = totallen  + len(wins[f])

    trainlen = int(totallen*percentage)
    validlen = totallen - trainlen

    sample = wins.values()[0][0]
    winlen = sample['end'] - sample['start']

    train_frags = np.empty([trainlen, 109, winlen])
    train_flags = np.empty([trainlen, 3])

    valid_frags = np.empty([validlen, 109, winlen])
    valid_flags = np.empty([trainlen, 3])

    counter = 0
    train_skipcount = 0
    valid_skipcount = 0

    for f in wins:
        filename = os.path.join(filedir, f)
        filepointer = h5py.File(filename, 'r')

        for window in wins[f]:

            # create image from window
            beg = int(window['start'])
            end = int(window['end'])

            scale = filepointer.attrs.get('scale', [0, 60])

            # Take the fragment of the right image
            fragment = filepointer[imageType][:, beg:end]

            if train_frags[0,:,:].shape != fragment.shape :
                if counter <= trainlen:
                    train_skipcount += 1
                else :
                    valid_skipcount += 1
                #Skip fragments with unexpected shape ( not frags[0,:,:].shape )
                continue;


            # create array of corresponding output flags
            flagment = np.array([1.0 if window['stressful'] else 0.0,
                1.0 if window['relaxing'] else 0.0,
                1.0 if window['sudden'] else 0.0])


            # add to train or validation set
            if counter < trainlen:
                train_frags[counter, :, :] = fragment
                train_flags[counter, :] = flagment
            else :
                valid_frags[counter - trainlen, :, :] = fragment
                valid_flags[counter - trainlen, :] = flagment

            counter += 1
            skipcount = train_skipcount + valid_skipcount

            totcount = counter + skipcount
            progress = (totcount / float(totallen))
            loadbar = '#' * int(round(20*progress)) +  ' ' * int(round(20*(1-progress)))

            print '\r[{0}] {1}% of {2} windows processed, {3} windows skipped'.format(loadbar,
                int(round(100*progress)),
                totallen,
                skipcount),
    print #this is a newline
    train_g = train_frags[:trainlen-train_skipcount, :]
    train_l = train_flags[:trainlen-train_skipcount, :]
    valid_g = valid_frags[:validlen-valid_skipcount, :]
    valid_l = valid_flags[:validlen-valid_skipcount, :]
    return train_g, train_l, valid_g, valid_l




def toImageDataWindowSplit(wins, percentage, filedir, imageType = 'energy'):
    if percentage > 1 or percentage < 0:
        RuntimeError('Percentage should be between 0 and 1')

    totallen = 0
    for f in wins:
        totallen  = totallen  + len(wins[f])

    trainlen = int(totallen*percentage)
    validlen = totallen - trainlen

    sample = wins.values()[0][0]
    winlen = sample['end'] - sample['start']

    train_frags = np.empty([trainlen, 109, winlen])
    train_flags = np.empty([trainlen, 3])

    valid_frags = np.empty([validlen, 109, winlen])
    valid_flags = np.empty([trainlen, 3])

    counter = 0
    train_skipcount = 0
    valid_skipcount = 0

    for f in wins:
        filename = os.path.join(filedir, f)
        filepointer = h5py.File(filename, 'r')

        for window in wins[f]:

            # create image from window
            beg = int(window['start'])
            end = int(window['end'])

            scale = filepointer.attrs.get('scale', [0, 60])

            fragment = filepointer['energy'][:, beg:end]
            if train_frags[0,:,:].shape != fragment.shape :
                if counter <= trainlen:
                    train_skipcount += 1
                else :
                    valid_skipcount += 1
                #Skip fragments with unexpected shape ( not frags[0,:,:].shape )
                continue;


            # create array of corresponding output flags
            flagment = np.array([1.0 if window['stressful'] else 0.0,
                1.0 if window['relaxing'] else 0.0,
                1.0 if window['sudden'] else 0.0])


            # add to train or validation set
            if counter < trainlen:
                train_frags[counter, :, :] = fragment
                train_flags[counter, :] = flagment
            else :
                valid_frags[counter - trainlen, :, :] = fragment
                valid_flags[counter - trainlen, :] = flagment

            counter += 1
            skipcount = train_skipcount + valid_skipcount

            totcount = counter + skipcount
            progress = (totcount / float(totallen))
            loadbar = '#' * int(round(20*progress)) +  ' ' * int(round(20*(1-progress)))

            print '\r[{0}] {1}% of {2} windows processed, {3} windows skipped'.format(loadbar,
                int(round(100*progress)),
                totallen,
                skipcount),
    print #this is a newline
    train_g = train_frags[:trainlen-train_skipcount, :]
    train_l = train_flags[:trainlen-train_skipcount, :]
    valid_g = valid_frags[:validlen-valid_skipcount, :]
    valid_l = valid_flags[:validlen-valid_skipcount, :]
    return train_g, train_l, valid_g, valid_l


#fig, ax = plt.subplots(figsize=(20,4))
#image = ax.imshow(filepointer['energy'][:,beg:end], aspect='auto', interpolation='none', origin='bottom')
#fig.colorbar(image)
#plt.show()
