#!/usr/bin/env python3
# ==============================================================================
# Authors: Carl Larsson, Pontus Svensson, Samuel Wågbrant
# Description: Main file used for running the classifier on the board
# Date: 2023-05-10
# ==============================================================================


import numpy as np
import argparse
import sys
import os
import time
import threading
import keyboard as kb
from collection.data_collection import load_data
from filtering.filter import bandpass, notch
from segmentation.segmentation import data_segmentation
from feature.feature import fe
from dimension.dimension import pca_func
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
    parser = argparse.ArgumentParser(prog="main_board.py", description="EMG finger movement classification on a microcontroller")

    parser.add_argument('-f', required=True, type=str, help="Enter filepath of pmml-model to load")
    parser.add_argument('--p', default=False, type=bool, help="Printing the classification, tested on beaglebone green")
    parser.add_argument('--w', default=False, type=bool, help="Write the classification and keyboard input to file, needs to be run as root")
    parser.add_argument('--hz', default=1000, type = int, help = "Set sampling rate")
    parser.add_argument('--ws', default=0.250, type = float, help = "Set window size (s)")
    parser.add_argument('--ol', default=0.125, type = float, help = "Set window overlap (s)")
    parser.add_argument('--nc', default=11, type = int, help = "Set number of classes")
    parser.add_argument('--pcanc', default=2, type = int, help = "Set number of PCA components")
    parser.add_argument('--nrs', default=377, type = int, help = "Set number of samples to load")

    args = parser.parse_args(argv)

    # One of the two modes has to be selected. Running on board or writing to file on computer
    if not (args.p or args.w):
        parser.error("Choose mode to run, --p True or --w True\nExiting...")

    # Load trained classifier
    time_to_load_model = time.time()
    print("Loading Model: ", args.f)
    model = load_model(args.f)
    print(f"Model loaded in: {time.time()-time_to_load_model:.1f}s")
    t1 = threading.Thread(target=readKey)
    t1.start()
    input("Press enter to start classification...")
            # Run classification on the microcontroller
    if args.p:
        time_to_make_classification = time.time()
        while (True):
            if (label == 11):
                break

            # Load data
            data = load_data(args.nrs)
            data = np.array(data)

            # Perform filtering
            data = bandpass(data, args.hz)
            data = notch(data, args.hz)

            # Perform segmentation
            segment_arr = data_segmentation(data, args.hz, args.ws, args.ol, args.nc)

            # Performs feature extraction
            segment_arr = fe(segment_arr, args.hz)

            # Perform PCA
            segment_arr = pca_func(segment_arr, args.pcanc)

            # The model predicts which class the data belongs to
            prediction = model.evaluateAll(segment_arr)
            print(prediction[['Integer labels']])
            print(f"Classification took: {time.time()-time_to_make_classification:.1f}s")
            time_to_make_classification = time.time()
    # Write to file on computer
    elif args.w:
        # Create thread to read keyboard
        print("Starting thread to read keyboard input...")
        t1 = threading.Thread(target=readKey)
        t1.start()
        header = ["Integer labels", "key"]
        # Write labels to a file on computer
        file_to_write = 'dataset.txt'
        print("Writing to file: ", file_to_write)
        with open(file_to_write, 'w') as f:
            while True:
                # Load data
                data = load_data(args.nrs)
                data = np.array(data)

                # Perform filtering
                data = bandpass(data, args.hz)
                data = notch(data, args.hz)

                # Perform segmentation
                segment_arr = data_segmentation(
                    data, args.hz, args.ws, args.ol, args.nc)

                # Performs feature extraction
                segment_arr = fe(segment_arr, args.hz)

                # Perform PCA
                segment_arr = pca_func(
                    segment_arr, args.pcanc)

                # The model predicts which class the data belongs to
                prediction = model.evaluateAll(segment_arr)
                prediction['key'] = label
                toPrint = prediction[header].to_string(
                    header=False, index=False, columns=header)
                if (label == 11):
                    break
                f.write(toPrint+'\n')
                f.flush()
        f.close()
    print("Ending program...")
    os._exit(os.EX_OK)
# ==============================================================================
if __name__ == "__main__":
    main(sys.argv[1:])
# ==============================================================================
