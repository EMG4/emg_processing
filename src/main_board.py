#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Main file used for running the classifier on the board
# Date: 2023-05-10
#==============================================================================


import numpy as np
import tensorflow as tf
import os
from segmentation.segmentation import data_segmentation
from feature.feature import fe
from dimension.dimension import pca_func
from sklearn.neighbors import KNeighborsClassifier


#==============================================================================
# Loads the data
def load_data(file_name):
    tf_model = 0
    
    if(tf_model):
        # Load data (TensorFlow)
        dir_path = os.path.join(os.getcwd(), "trained_models")
        model = tf.saved_model.load(dir_path)
        return model
    else:
        # Load data (scikit learn)
        file = open(file_name, 'rb')
        classifier = pickle.load(file)
        return classifier
#==============================================================================
# Main function
def main(argv):
    file_name = "trained_classifier.txt"
    sampling_frequency = 1000
    window_size = 0.25
    overlap = 0.125
    number_classes = 11
    number_principal_components = 5

    # Load data

    # Load the model from file
    model = load_data(file_name)

    # Perform segmentation
    segment_arr, label_arr = data_segmentation(data, sampling_frequency, window_size, overlap, number_classes)

    # Performs feature extraction
    segment_arr = fe(segment_arr, sampling_frequency)

    # Perform PCA
    segment_arr = pca_func(segment_arr, number_principal_components)

    prediction = model.predict(segment_arr)

    print(prediction)
#==============================================================================
if __name__ == "__main__":
    main(sys.argv[1:])
#==============================================================================

