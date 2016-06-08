
import sys
import os
import pickle

import numpy as np

label_dir = os.path.join(os.getcwd(), '../CNN')
sys.path.insert(0, label_dir)

import cnn

import matplotlib.pyplot as plt
import math
from scipy import interpolate
from scipy import stats
from scipy.ndimage.filters import gaussian_filter1d
def save_fmap() :
    import h5py
    
    label_dir = os.path.join(os.getcwd(), '../labeling')
    #load the windows from file
    windows = pickle.load(open(os.path.join(label_dir, 'labels.pickle'), 'rb'))

    #cut out the actual cochleogram fragments 
    # and generate a list of corresponding flags (stressfull, relaxing, sudden)
    codedir, dummy = os.path.split(os.getcwd())
    relacsdir, dummy = os.path.split(codedir)
    hdf5_path = os.path.join(relacsdir, 'sound_files/hdf5')

    first_file = windows.keys()[0]
    filename = os.path.join(hdf5_path, first_file)
    filepointer = h5py.File(filename, 'r')  
    fmap = np.asarray(filepointer.attrs['fMap'], dtype = float)
    ptnsplit = (map(int, filepointer.attrs['ptnsplit'].translate(None, '[]').split(',')))

    print filepointer.attrs.keys()

    same = True
    for f in windows:

        filename = os.path.join(hdf5_path, f)
        filepointer = h5py.File(filename, 'r')

        new_fmap = np.asarray(filepointer.attrs['fMap'], dtype = float)
        new_ptnsplit = (map(int, filepointer.attrs['ptnsplit'].translate(None, '[]').split(',')))
            
        if not np.all(fmap == new_fmap) :
            print 'Not all frequency maps are the same: '
            print fmap 
            print new_fmap

            same = False
            break

        if not np.all(ptnsplit == new_ptnsplit) :
            print 'Not all frequency maps are the same: '
            print ptnsplit 
            print new_ptnsplit

            same = False
            break

    if same :
        with open(os.path.join(os.getcwd(), 'fmap.np'), 'wb') as f:
            print len(fmap)
            print len(ptnsplit)
            np.save(f, fmap[ptnsplit])
        print 'Frequency map saved successfully'

def load_fmap() :
    with open(os.path.join(os.getcwd(), 'fmap.np'), 'rb') as f:
        fmap = np.load(f)
    print 'Frequency map loaded'

    return fmap
        
def bin_centers(bin_edges) :
    centers = np.empty(len(bin_edges)-1)
    
    for it in range(len(bin_edges)-1):
        prev_edge = bin_edges[it]
        next_edge = bin_edges[it+1]
        centers[it] = (prev_edge + next_edge)/2

    return centers

def all_distances(ordered_list) :
    if len(ordered_list) < 2 :
        return []
    
    return_list = []
    head = ordered_list[0]
    tail = ordered_list[1:]
    for tail_part in tail :
        return_list = np.append(return_list, tail_part - head)

    return np.append(return_list, all_distances(tail))

def harmonics(ordered_freqs, precision = 0.1, only_peak_sequences = True) :
    precision /= 2
    #jagged matrix of indices: [[0,1,2,3],[1,2,3],[2,3]]
    peak_locs = [list(range(x, len(ordered_freqs))) for x in xrange(len(ordered_freqs)-1)]

    diffs = []
    base_freqs = []
    conseq_dict = {}
    #Note: peak here means the index that corresponds to a frequency in ordered_freqs
    for peak_list_idx in range(len(peak_locs)) :
        first_peak_freq = ordered_freqs[peak_locs[peak_list_idx][0]]
        current_peak_list_length = len(peak_locs[peak_list_idx])
        for scnd_peak_idx in range(1, current_peak_list_length) :
            scnd_peak_freq = ordered_freqs[peak_locs[peak_list_idx][scnd_peak_idx]]
            diff = scnd_peak_freq - first_peak_freq
            # Index of the previous frequency on which a peak has been found with constant spacing.
            prev_peak_idx = 0
            prev_peak = peak_locs[peak_list_idx][prev_peak_idx]
            previous_n = 0

            for new_peak_idx in range(scnd_peak_idx, current_peak_list_length) :
                new_peak = peak_locs[peak_list_idx][new_peak_idx]
                new_peak_freq = ordered_freqs[new_peak]
                large_diff = new_peak_freq - first_peak_freq
                # n is a measure for how precise large_diff is a multiple of diff
                n = large_diff/diff
                
                actual_diff = ordered_freqs[new_peak] - ordered_freqs[prev_peak] 
                
                if (n%1 > 1 - precision or n%1 < precision) and round(n) == previous_n + 1: #if it is about a multiple (precision% margin)
                    
                    previous_n = round(n)
                    if n != 1. or not only_peak_sequences:
                        diffs = np.append(diffs, actual_diff)
                        base_freqs = np.append(base_freqs, first_peak_freq)
                        if not conseq_dict.has_key(first_peak_freq):
                            conseq_dict[first_peak_freq] = {}
                        if not conseq_dict[first_peak_freq].has_key(diff):
                            conseq_dict[first_peak_freq][diff] = 0
                            if only_peak_sequences: 
                                conseq_dict[first_peak_freq][diff] += 1
                        conseq_dict[first_peak_freq][diff] += 1

                    #We now want to remove new_peak_idx from the list that has prev_peak_idx as its head, because
                    #otherwise we will save this distance twice.  
                    prev_as_head_peak_list_index = prev_peak
                    #Check so that we don't remove anything from the list we are currently iterating over
                    if prev_as_head_peak_list_index != peak_list_idx and new_peak in peak_locs[prev_as_head_peak_list_index]:
                        print 'Removed', ordered_freqs[new_peak], 'from', prev_as_head_peak_list_index
                        peak_locs[prev_as_head_peak_list_index].remove(new_peak)
                    
                    prev_peak = new_peak
                else :
                    if actual_diff > diff :
                        round(n)
                        break

    return diffs, base_freqs, conseq_dict



def gaussian_kernel(length):
    mu = length/2.0
    sig = length/6
    x = np.asarray(range(length))
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

if __name__ == '__main__':
    save_fmap()

    quit()
    fmap = load_fmap()
    
    plt.ion()
    plt.figure()
    plt.show()
    plt.plot(fmap)
    plt.draw()
    
    X_train, Y_train, X_test, Y_test = cnn.load_data()

    centers = bin_centers(fmap)

    window_to_inspect = 146
    time_step_to_inspect = 24
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[3])
    X_train = X_train.transpose(0, 2, 1)
    for win_idx, win in enumerate(X_train):
        if win_idx == window_to_inspect or (Y_train[win_idx, 4] and window_to_inspect == -1): #If Human sound
            plt.figure()
            plt.show()
            plt.imshow(win.transpose(1,0))
            plt.draw()
            if window_to_inspect == -1 :
                print win_idx
                user_input = raw_input("Do you like this figure? (y/n): ")
                if user_input == 'n':
                    plt.close()
                    continue

            plt.ioff()

            for time_step_idx, time_step in enumerate(win[5:-5]):
                if time_step_to_inspect == -1 or time_step_to_inspect == time_step_idx  :
                    s = 0

                    smoothed_time_step = np.sum(win[time_step_idx-s:time_step_idx+1+s], axis = 0)
                    f = interpolate.interp1d(centers, smoothed_time_step[::-1], 'cubic')
                    x = np.arange(centers[0], centers[-1], 5)
                    fx = f(x)
                    peaks = np.asarray([])
                    peak_locations = np.asarray([])
                    for it in range(1, len(fx)-1) :
                        if fx[it-1] < fx[it] and fx[it] > fx[it+1] :
                            peaks = np.append(peaks, fx[it])
                            peak_locations = np.append(peak_locations, x[it])

                    plt.figure()

                    plt.subplot(311)
                    plt.plot(centers, smoothed_time_step[::-1], 'b')
                    plt.plot(x, fx, 'g')
                    plt.plot(peak_locations, peaks, 'ro')
                    plt.xlabel('Frequency')
                    plt.ylabel('Intensity')
                    

                    
                    # x = range(int(max(dists)))
                    x = range(2000)
                    
                    dists, base_freqs, conseq_dict = harmonics(peak_locations, only_peak_sequences=False)
                    freq_set = set(base_freqs)

                    def get_density(dists, kernel_size, step_size, max_freq) :
                        no_bins = int(round(max_freq/float(step_size)))
                        hist, bin_edges = np.histogram(dists, bins = no_bins, range = (0,max_freq))
                        if kernel_size != None:
                            hist = np.convolve(hist, gaussian_kernel(kernel_size), 'same')

                        return hist, bin_centers(bin_edges)
                    def plot_densities(kernel_size = None) :
                        for f in freq_set:
                            f_dists = dists[base_freqs == f]
                            density, freqs = get_density(f_dists, kernel_size, 1, 2000)
                            plt.plot(freqs, density)

                    

                    plt.subplot(312)
                    for c_dict in conseq_dict.values():
                        plt.plot(c_dict.keys(), c_dict.values(), 'o')
                    axes = plt.gca()
                    axes.set_xlim([0,2000])
                    axes.set_ylim([-0.2,3.2])
                    plt.ylabel('Number of consequtive peaks')
                    plt.xlabel('Frequency difference between two peaks')
                    #plot_densities(10)

                    
                    plt.subplot(313)
                    density, freqs = get_density(dists, 50, 1, 2000)
                    plt.plot(freqs, density)
                    axes = plt.gca()
                    axes.set_xlim([0,2000])
                    plt.ylabel('Peak interval density')
                    plt.xlabel('Frequency difference between two peaks')

                    plt.tight_layout() 
                    plt.show()

                


