#!/usr/bin/env python3
# ==============================================================================
# Authors: Carl Larsson, Pontus Svensson, Samuel Wågbrant
# Description: Main file used for running the classifier on the board
# Date: 2023-05-10
# ==============================================================================
import numpy as np
import argparse
import sys
# import tensorflow as tf
# import os
import time
import threading
import keyboard as kb
# from pandas.io.pickle import pc
from collection.data_collection import load_data
from filtering.filter import bandpass, notch
from segmentation.segmentation import data_segmentation
from feature.feature import fe
from dimension.dimension import pca_func
# from sklearn.neighbors import KNeighborsClassifier
from jpmml_evaluator import make_evaluator
# ==============================================================================
# Function to read the keyboard inputs


def readKey():
    global label
    while True:
        key = kb.read_key()
        if key == 'q':
            label = 1
        elif key == 'a':
            label = 6
        elif key == 'w':
            label = 2
        elif key == 's':
            label = 7
        elif key == 'e':
            label = 3
        elif key == 'd':
            label = 8
        elif key == 'r':
            label = 4
        elif key == 'f':
            label = 9
        elif key == 't':
            label = 5
        elif key == 'g':
            label = 10
        elif key == 'c':
            label = 11
        else:
            label = 0
# ==============================================================================
# Loads the trained model
def load_model(file_name):
    tf_model = 0
    if (tf_model):
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

# ==============================================================================
# Main function
def main(argv):
    parser = argparse.ArgumentParser(
        prog="main.py", description="EMG finger movement classification")
    parser.add_argument('-f', required=True, type=str,
                        help="Enter filepath of pmml-model to load")

    parser.add_argument('--p', default=False, type=bool,
                        help="Printing the classification, tested on beaglebone green")

    parser.add_argument('--w', default=False, type=bool,
                        help="Write the classification and keyboard input to file, needs to be run as root")

    args = parser.parse_args(argv)
    if not (args.p or not args.w):
        parser.error("Choose mode to run, -p or -w\nExiting...")

    file_name = args.f
    sampling_frequency = 1000
    window_size = 0.25
    overlap = 0.125
    number_classes = 11
    number_principal_components = 2
    time_to_load_model = time.time()
    print("Loading Model: ", file_name)
    model = load_model(file_name)
    print(f"Model loaded in: {time.time()-time_to_load_model:.1f}s")
    input("Press enter to start classification...")

    if args.p:

        time_to_make_classification = time.time()
        while (True):
            # Load data
            data = load_data()
            data = np.array(data)

            # Perform filtering
            data = bandpass(data, sampling_frequency)
            data = notch(data, sampling_frequency)

            # Perform segmentation
            segment_arr = data_segmentation(
                data, sampling_frequency, window_size, overlap, number_classes)

            # Performs feature extraction
            segment_arr = fe(segment_arr, sampling_frequency)

            # Perform PCA
            segment_arr = pca_func(segment_arr, number_principal_components)

            # The model predicts which class the data belongs to
            prediction = model.evaluateAll(segment_arr)
            print(prediction[['Integer labels']])
            print("Classification took: " +
                  str(time.time()-time_to_make_classification))
            time_to_make_classification = time.time()

    elif args.w:
        # Load the model from file
        print("Starting thread to read keyboard input...")
        t1 = threading.Thread(target=readKey)
        t1.start()
        header = ["Integer labels", "key"]
        file_to_write = 'dataset.txt'
        print("Writing to file", file_to_write)
        with open(file_to_write, 'w') as f:
            while True:
                # Load data
                data = load_data()
                data = np.array(data)

                # Perform filtering
                data = bandpass(data, sampling_frequency)
                data = notch(data, sampling_frequency)

                # Perform segmentation
                segment_arr = data_segmentation(
                    data, sampling_frequency, window_size, overlap, number_classes)

                # Performs feature extraction
                segment_arr = fe(segment_arr, sampling_frequency)

                # Perform PCA
                segment_arr = pca_func(
                    segment_arr, number_principal_components)

                # The model predicts which class the data belongs to
                prediction = model.evaluateAll(segment_arr)
                # print(prediction[['Integer labels']])
                prediction['key'] = label
                toPrint = prediction[header].to_string(
                    header=False, index=False, columns=header)
                # print(str(prediction[['Integer labels']]))
                # print(toPrint)
                if (label == 11):
                    break
                f.write(toPrint+'\n')
                f.flush()
        t1.join()
        f.close()
# ==============================================================================
if __name__ == "__main__":
    main(sys.argv[1:])
# ==============================================================================
