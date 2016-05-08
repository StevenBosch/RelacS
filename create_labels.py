#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import yaml

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

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage:", sys.argv[0], "<window definitions.yaml>")
        sys.exit(0)

    with open(sys.argv[1], 'r') as f:
        windows = yaml.load(f)
        print(windows)

    samples = []
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
                samples.append(row)
            except (ValueError, IndexError) as e:
                # Ignore lines that are not proper entries
                print('Found an incorrect entry:', line)

    samples.sort(key=lambda s: s[FILENAME].lower() + str(s[START_TIME]))
    print("Found", len(samples), "samples")

    for sample in samples:
        for window in windows['windows']:
            time = sample[START_TIME]
            while time + window['size'] <= sample[END_TIME]:
                time += window['stride']
                print("Generating window for file", sample[FILENAME], "at", time)
