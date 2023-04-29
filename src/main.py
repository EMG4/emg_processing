#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Performs dimensionallity reduction on input file and provides output file
# Date: 2023-04-25
#==============================================================================


import sys
import argparse
import numpy as np
from filtering.filter import rm_offset, bandpass, notch
from segmentation.segmentation import segmentation
from feature.feature import fe
from dimension.dimension import pca_func, ofnda_func
from classifier.classifier import lda, mlp, ann, cnn


#==============================================================================
# Loads the data
def load_data(file_name):
    # Load data
    data = np.loadtxt(file_name, delimiter='\t')

    return data
#==============================================================================
# Main function
def main(argv):

    parser = argparse.ArgumentParser(prog = "main.py", description = "EMG finger movement identification")

    # Options
    parser.add_argument('--ro', default=False, type = bool, help = "Run remove dc offset")
    parser.add_argument('--rb', default=False, type = bool, help = "Run apply bandpass filter")
    parser.add_argument('--rn', default=False, type = bool, help = "Run apply notch filter")
    parser.add_argument('--rpca', default=False, type = bool, help = "Run apply PCA")
    parser.add_argument('--rofnda', default=False, type = bool, help = "Run apply OFNDA")
    parser.add_argument('--rlda', default=False, type = bool, help = "Run LDA")
    parser.add_argument('--rmlp', default=False, type = bool, help = "Run MLP")
    parser.add_argument('--rann', default=False, type = bool, help = "Run ANN")
    parser.add_argument('--rcnn', default=False, type = bool, help = "Run CNN")
    # Segmentation parameters
    parser.add_argument('--hz', default=1000, type = int, help = "Set sampling rate")
    parser.add_argument('--ws', default=0.250, type = int, help = "Set window size (s)")
    parser.add_argument('--ol', default=0.100, type = int, help = "Set window overlap (s)")
    parser.add_argument('--nc', default=11, type = int, help = "Set number of classes")
    # Dimension Reduction parameters 
    parser.add_argument('--pcanc', default=1, type = int, help = "Set amount of PCA components")
    # Multiple classifier parameters
    parser.add_argument('--tsp', default=0.8, type = float, help = "Set training set proportions")
    parser.add_argument('-i', default=100, type = int, help = "Set number of iterations")
    parser.add_argument('-l', default=3, type = int, help = "Set number of layers")
    parser.add_argument('--af', default='relu', type = str, help = "Set activation function")
    parser.add_argument('-a', default=0.1, type = float, help = "Set learning rate, alpha")
    # LDA parameters
    parser.add_argument('--ldanc', default=1, type = int, help = "Set amount of LDA components")
    # MLP parameters
    parser.add_argument('--sf', default='adam', type = str, help = "Set solver function")
    parser.add_argument('--lrm', default='constant', type = str, help = "Set learning rate model")
    # ANN parameters
    parser.add_argument('-k', default=5, type = int, help = "Set k for k fold cross validation")
    parser.add_argument('--dr', default=0.2, type = float, help = "Set dropout rate")
    parser.add_argument('-n', default=20, type = int, help = "Set number of neurons")
    parser.add_argument('-mn', default=3, type = int, help = "Set maximum norm of the weights")


    args = parser.parse_args(argv)


    layer_arr = [(15, "relu", 3), (15, "relu", 3), (15, "relu", 3)]


    # Load data (csv)
    data_np_arr = load_data("test.txt")
    # First column is the data
    raw_data = data_np_arr[:, 0]
    # Second column is labels
    labels = data_np_arr[:, 1]


    # Perform preprocessing
    # Remove dc offset
    if(args.ro):
        raw_data = rm_offset(raw_data, args.hz)
    # Apply bandpass filter
    if(args.rb):
        raw_data = bandpass(raw_data, args.hz)
    # Apply nothc filter
    if(args.rn):
        raw_data = notch()
    

    # Perform segmentation
    segment_arr, label_arr = segmentation(raw_data, labels, args.hz, args.ws, args.ol, args.nc)


    # Performs feature extraction
    segment_arr = fe(segment_arr, args.hz)


    # Chooses dimension reduction
    if(args.rpca):
        segment_arr = pca_func(segment_arr, args.pcanc)
    elif(args.rofnda):
        segment_arr = ofnda_func()
    else:
        segment_arr = segment_arr


    # Chooses classifier
    if(args.rlda):
        classifier = lda(segment_arr, label_arr, args.tsp, args.ldanc)
    elif(args.rmlp):
        classifier = mlp(segment_arr, label_arr, args.tsp, args.l, args.af, args.sf, args.lrm, args.a, args.i)
    elif(args.rann):
        classifier = ann(segment_arr, label_arr, args.k, args.dr, input_dim, args.l, args.a, args.i, args.af, args.n, args.mn)
    elif(args.rcnn):
        classifier = cnn() 
    else:
        print("no classifier is chosen")
        exit()


    # Saves the trained classifier to a json file


#==============================================================================
if __name__ == "__main__":
    main(sys.argv[1:])
