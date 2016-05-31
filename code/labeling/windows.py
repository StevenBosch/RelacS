#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import yaml
import h5py
import numpy as np
import os
import os.path
import math
import pickle
from create_labels import replace_last_two

ID         = 0
FILENAME   = 1
LABELER    = 2
START_TIME = 3
END_TIME  = 4
STRESSFUL  = 5
RELAXING   = 6
SUDDEN     = 7
CATEGORY   = 8
OTHER      = 9
NO = ('no', '0')

def print_windows(windows):
    for f in windows:
        print(f)
        for w in windows[f]:
            print(w['start'], w['end'], w['stressful'], w['relaxing'], w['sudden'], w['category'])

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:", sys.argv[0], "<window definitions.yaml>")
        sys.exit(-1)

    with open(sys.argv[1], 'r') as f:
        window_defs = yaml.load(f)
        print(window_defs)

    data = []
    with open("labeling.csv", 'r') as f:
        for line in f.readlines():
            row = line.strip().split(',')
            try:
                row[ID] = int(row[ID])
                if len(row[FILENAME]) == 0:
                    continue # Skip entries without a filename
                row[START_TIME] = float(row[START_TIME].split(':')[-1])
                row[END_TIME] = float(row[END_TIME].split(':')[-1])
                row[STRESSFUL] = row[STRESSFUL].lower() not in NO
                row[RELAXING]  = row[RELAXING].lower() not in NO
                row[SUDDEN]    = row[SUDDEN].lower() not in NO
                data.append(row)
            except (ValueError, IndexError) as e:
                # Ignore lines that are not proper entries
                print('Found an incorrect entry:', line.strip())

    print("Found", len(data), "samples")

    # Split the data into files
    files = {}
    for d in data:
        if d[FILENAME] not in files.keys():
            files[d[FILENAME]] = {'ranges': [], 'id': d[ID]}
        files[d[FILENAME]]['ranges'].append(d)
    for f in files:
        files[f]['ranges'].sort(key=lambda k:[START_TIME])

    filedir = '../../sound_files/hdf5/'

    print("Processing files...")
    windows = {}
    for f in files:
        # Read HDF5 file
        fname = f + '.wav.2.hdf5'
        filename = os.path.join(filedir, fname)

        # Deal with weird filenames
        if not os.path.isfile(filename):
            fname = replace_last_two(fname, '-', ':')
            filename = os.path.join(filedir, fname)
            if not os.path.isfile(filename):
                print("Couldn't open a file:", filename)
                continue

        filepointer = h5py.File(filename, 'r')
        windows[fname] = []

        # Calculate sample rate
        attrs = filepointer.attrs
        length = filepointer['energy'].shape[1]
        fs = length / attrs['duration']
        scale = int(fs/190 + 0.1)

        # Find how much to skip at the start
        energy = filepointer['energy']
        start = 0
        while max(energy[:,start].flatten()) == 0:
            start += 1

        # Create the labels for each window
        for wdef in window_defs['windows']:
            size   = int(wdef['size'])
            stride = int(wdef['stride'])
            for w in range(start, length-size*scale, stride*scale):
                stressful = False
                relaxing = False
                sudden = False
                categories = set()
                for r in files[f]['ranges']:
                    start = math.floor(r[START_TIME] * fs)
                    end = math.floor(r[END_TIME] * fs)
                    # Check if the marked range fits exactly in the window
                    # If it does than we want to copy its label
                    if start > w and end < w+stride*scale:
                        stressful = r[STRESSFUL]
                        relaxing  = r[RELAXING]
                        sudden    = r[SUDDEN]
                        categories = [r[CATEGORY]]
                        break
                    # Check if the marked range overlaps with the window
                    if (start < w and end > w) or \
                            (start < w+stride*scale and end > w+stride*scale):
                        stressful = stressful or r[STRESSFUL]
                        relaxing  = relaxing  or r[RELAXING]
                        sudden    = relaxing  or r[SUDDEN]
                        categories.add(r[CATEGORY])

                windows[fname].append({'start': w,
                    'end': w + int(wdef['size']) * scale,
                    'stressful': stressful,
                    'relaxing': (relaxing and not stressful),
                    'sudden': sudden,
                    'category': list(categories),})

        # Save the windows
        with open('labels.pickle', 'wb') as f:
            pickle.dump(windows, f)

        #print_windows(windows)
