#!/bin/env python3
import argparse
from os import listdir, path
from random import shuffle

parser = argparse.ArgumentParser(
    description="Returns percentage of matching lines from two files.")
parser.add_argument('file1', type=str) # test file
parser.add_argument('file2', type=str) # predicted file
args = parser.parse_args()

with open(args.file1, 'r') as f:
    lines_1 = f.read().splitlines()

with open(args.file2, 'r') as f:
    lines_2 = f.read().splitlines()

assert len(lines_1) == len(lines_2)

matching = [1 if lines_1[i] == lines_2[i] else 0 for i in range(len(lines_1))]
accuracy = sum(matching) / len(matching)
print("Accuracy: {:.2f}%".format(accuracy * 100))

