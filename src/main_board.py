#!/usr/bin/env python3
#==============================================================================
# Authors: Carl Larsson, Pontus Svensson, Samuel Wågbrant
# Description: Main file used for running the classifier on the board
# Date: 2023-05-10
#==============================================================================


import numpy as np
#import tensorflow as tf
# import os
import time
from collection.data_collection import load_data
from filtering.filter import bandpass, notch
from segmentation.segmentation import data_segmentation
from feature.feature import fe
from dimension.dimension import pca_func
# from sklearn.neighbors import KNeighborsClassifier
from jpmml_evaluator import make_evaluator


#==============================================================================
# Loads the trained model
def load_model(file_name):
    tf_model = 0
    if(tf_model):
        # Load data (TensorFlow)
        '''
        dir_path = os.path.join(os.getcwd(), "trained_models")
        model = tf.saved_model.load(dir_path)
        return model
        '''
    else:
        # Load data (scikit learn)
        clf = make_evaluator(file_name).verify()
        return clf
#==============================================================================
# Main function
def main():
    file_name = "./trained_scikit_models/svm.pmml"
    sampling_frequency = 1000
    window_size = 0.25
    overlap = 0.125
    number_classes = 11
    number_principal_components = 2

    time_to_load_model = time.time()
    # Load the model from file
    print("Loading Model")
    model = load_model(file_name)
    print("Model loaded in: "+str(time.time()-time_to_load_model))

    input("Press enter to start classification...")

    time_to_make_classification = time.time()
    while(True):
        # Load data
        data = load_data()
        data = np.array(data)

        # Perform filtering
        data = bandpass(data, sampling_frequency)
        data = notch(data, sampling_frequency)

        # Perform segmentation
        segment_arr = data_segmentation(data, sampling_frequency, window_size, overlap, number_classes)

        # Performs feature extraction
        segment_arr = fe(segment_arr, sampling_frequency)

        # Perform PCA
        segment_arr = pca_func(segment_arr, number_principal_components)

        # The model predicts which class the data belongs to
        prediction = model.evaluateAll(segment_arr)
        print(prediction[['Integer labels']])
        print("Classification took: "+str(time.time()-time_to_make_classification)
        time_to_make_classification = time.time()
#==============================================================================
if __name__ == "__main__":
    main()
#==============================================================================

