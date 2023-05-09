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
from classifier.classifier import lda, mlp, ann, xgboost_classifier


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
    parser.add_argument('-f', required = True, type = str, help = "Choose input file")
    parser.add_argument('--ro', default=False, type = bool, help = "Run remove dc offset")
    parser.add_argument('--rb', default=False, type = bool, help = "Run apply bandpass filter")
    parser.add_argument('--rn', default=False, type = bool, help = "Run apply notch filter")
    parser.add_argument('--rpca', default=True, type = bool, help = "Run apply PCA")
    parser.add_argument('--rofnda', default=False, type = bool, help = "Run apply OFNDA")
    parser.add_argument('--rlda', default=False, type = bool, help = "Run LDA")
    parser.add_argument('--rmlp', default=False, type = bool, help = "Run MLP")
    parser.add_argument('--rann', default=False, type = bool, help = "Run ANN")
    parser.add_argument('--rxgb', default=False, type = bool, help = "Run XGBoost")
    # Segmentation parameters
    parser.add_argument('--hz', required = True, type = int, help = "Set sampling rate")
    parser.add_argument('--ws', default=0.250, type = int, help = "Set window size (s)")
    parser.add_argument('--ol', default=0.125, type = int, help = "Set window overlap (s)")
    parser.add_argument('--nc', default=11, type = int, help = "Set number of classes")
    # Dimension Reduction parameters 
    parser.add_argument('--pcanc', default=2, type = int, help = "Set number of PCA components")
    # Multiple classifier parameters
    parser.add_argument('--tsp', default=0.8, type = float, help = "Set training set proportions (between 0 and 1)")
    parser.add_argument('-k', default=5, type = int, help = "Set k for k fold cross validation")
    parser.add_argument('-i', default=10000, type = int, help = "Set number of iterations")
    parser.add_argument('-l', default=7, type = int, help = "Set number of layers")
    parser.add_argument('--af', default='relu', type = str, help = "Set activation function: tanh, relu")
    parser.add_argument('--sf', default='adam', type = str, help = "Set solver function: sgd, adam")
    parser.add_argument('-a', default=0.01, type = float, help = "Set learning rate, alpha (between 0 and 1)")
    # LDA parameters
    parser.add_argument('--ldanc', default=None, help = "Set number of LDA components")
    parser.add_argument('--ls', default="eigen", help = "Set LDA solver: svd, lsqr, eigen")
    # MLP parameters
    parser.add_argument('--lrm', default='constant', type = str, help = "Set learning rate model: constant, invscaling, adaptive")
    # ANN parameters
    parser.add_argument('-n', default=10, type = int, help = "Set number of neurons")
    parser.add_argument('--bs', default=10, type = int, help = "Set batch size")
    parser.add_argument('--dr', default=0.1, type = float, help = "Set dropout rate")
    # GA parameters
    parser.add_argument('--ns', default=100, type = int, help = "Set number of solutions in the population")
    parser.add_argument('--ng', default=200, type = int, help = "Set number of generations")
    parser.add_argument('--npm', default=40, type = int, help = "Set number of parents mating")

    args = parser.parse_args(argv)


    # Load data
    data_np_arr = load_data(args.f)
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
    # Apply notch filter
    if(args.rn):
        raw_data = notch(raw_data, args.hz)
    

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
        classifier = lda(segment_arr, label_arr, args.ldanc, args.k, args.ls)
    elif(args.rmlp):
        classifier = mlp(segment_arr, label_arr, args.l, args.af, args.sf, args.lrm, args.a, args.i, args.k, args.tsp)
    elif(args.rann):
        # Need input dim for the ANN input layer
        input_dim = segment_arr.shape[1]
        classifier = ann(segment_arr, label_arr, args.k, args.dr, input_dim, args.l, args.sf, args.i, args.af, args.n, args.bs, args.nc, args.ns, args.ng, args.npm)
    elif(args.rxgb):
        classifier = xgboost_classifier(segment_arr, label_arr, args.tsp, args.k, args.nc)
    else:
        print("no classifier is chosen")
        exit()


    # Saves the trained classifier to a json file


#==============================================================================
if __name__ == "__main__":
    main(sys.argv[1:])
