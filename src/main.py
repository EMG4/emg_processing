#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Performs dimensionallity reduction on input file and provides output file
# Date: 2023-04-25
#==============================================================================


import pandas as pd
from feature import fe
from dimension import pca_func, ofnda_func
from classifier import lda, mlp, cnn


#==============================================================================
# Loads the data
def load_data(file_name):
    # Load data
    data = np.loadtxt(file_name, sep='\t')

    return data
#==============================================================================
# Main function
def main():
    pca_number_components = 7 
    lda_number_components = 10
    layer_arr = [(15, "relu", 3), (15, "relu", 3), (15, "relu", 3)]


    # Load data (csv)
    data_np_arr = load_data("test.txt")
    # First column is the data
    raw_data = data_np_arr[:, 0]
    # Second column is labels
    labels = data_np_arr[:, 1]


    # Perform preprocessing
    # Remove dc offset
    if(run_rm_offset):
        raw_data = rm_offset(raw_data, sampling_rate)
    # Apply bandpass filter
    if(run_bandpass):
        raw_data = bandpass(raw_data, sampling_rate)
    # Apply nothc filter
    if(run_notch):
        raw_data = notch()
    

    # Perform segmentation
    if(run_segmentation):
        segment_arr, label_arr = segmentation(raw_data, labels, sampling_rate, window_size, overlap, num_classes):


    # Performs feature extraction
    segment_arr = fe(segment_arr)


    # Chooses dimension reduction
    if(run_pca)
        segment_arr = pca_func(segment_arr, pca_number_components)
    elif(run_ofnda)
        segment_arr = ofnda_func()
    else
        segment_arr = segment_arr


    # Chooses classifier
    if(run_lda)
        classifier = lda(segment_arr, label_arr, traing_set_proportion, lda_number_components)
    elif(run_mlp)
        classifier = mlp(segment_arr, label_arr, traing_set_proportion, layers, activation_func, solver_func, learning_rate_model, iterations)
    elif(run_ann)
        classifier = ann(segment_arr, label_arr, k_fold_splits, dropout_rate, input_dim, layer_arr, learning_rate, iterations)
    elif(run_cnn)
        classifier = cnn() 
    else
        print("no classifier is chosen")
        exit()


    # Saves the trained classifier to a json file


#==============================================================================
if __name__ == "__main__"":
    main()
