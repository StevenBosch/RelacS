#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import yaml
import h5py
import numpy as np
import os
import math
import pickle

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

def print_windows(data):
    for f in data:
        print('{}:'.format(f))
        for window in data[f]:
            print('\t{}\t{}\t{}\t{}\t{}'.format(window['start'], window['end'],
                'Stressful' if window['stressful'] else '\t',
                'Relaxing' if window['relaxing'] else '\t',
                'Sudden' if window['sudden'] else ''))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage:", sys.argv[0], "<window definitions.yaml>")
        sys.exit(0)

    with open(sys.argv[1], 'r') as f:
        windows = yaml.load(f)
        print(windows)

    data = []
    with open("labeling.csv", 'r') as f:
        for line in f.readlines():
            line = line.strip()
            row = line.split(',')
            try:
                row[ID] = int(row[ID])
                if len(row[FILENAME]) == 0:
                    continue # Skip files without a filename
                row[START_TIME] = float(row[START_TIME].split(':')[-1])
                row[END_TIME] = float(row[END_TIME].split(':')[-1])
                row[STRESSFUL] = False if row[STRESSFUL].lower() == 'no' else True
                row[RELAXING]  = False if row[RELAXING].lower() == 'no' else True
                row[SUDDEN]    = False if row[SUDDEN].lower() == 'no' else True
                data.append(row)
            except (ValueError, IndexError) as e:
                # Ignore lines that are not proper entries
                print('Found an incorrect entry:', line)

    print("Found", len(data), "samples")
    samples = {}
    # Find all the ranges in each file
    for d in data:
        if d[FILENAME] not in samples.keys():
            samples[d[FILENAME]] = {'ranges': [], 'id': d[ID]}
        samples[d[FILENAME]]['ranges'].append((d[START_TIME], d[END_TIME],
            d[STRESSFUL], d[RELAXING], d[SUDDEN]))
    # Sort the windows for each file by start time
    for s in samples:
        samples[s]['ranges'].sort(key=lambda k: k[0])

    filedir = '../../sound_files/hdf5'
    print(filedir)
    positive_windows = {}
    for sample in samples:
        positive_windows[sample] = []
        # Read HDF5 file
        filename = os.path.join(filedir, sample + '.wav.2.hdf5')
        try:
            filepointer = h5py.File(filename, 'r')
        except OSError as e:
            print("Couldn't open a file:", e)
            continue
        print('Processing', sample)

        # Calculate sample rate
        attrs = filepointer.attrs
        fs = filepointer['energy'].shape[1] / attrs['duration']

        # Create the actual (positive) windows
        for window in windows['windows']:
            for r in samples[sample]['ranges']:
                start_time = math.floor(r[0] * fs)
                end_time = math.ceil(r[1] * fs)
                if end_time - start_time < window['size']:
                    continue # Skip ranges where we can extract no windows
                # add windows
                for w in range(start_time, end_time-window['size'],
                        window['stride']):
                    positive_windows[sample].append({'start': w,
                        'end': w+window['size'], 'stressful': r[2],
                        'relaxing': r[3], 'sudden': r[4]})
                # Add a last window that ends on the last sample
                positive_windows[sample].append({
                    'start': end_time-window['size'], 'end': end_time,
                    'stressful': r[2], 'relaxing': r[3], 'sudden': r[4]})

    #print_windows(positive_windows)

    # Save the windows
    with open('labels.pickle', 'wb') as f:
        pickle.dump(positive_windows, f)
