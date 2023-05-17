#!/usr/bin/env python3
#==============================================================================
# Authors: Carl Larsson, Pontus Svensson, Samuel Wågbrant
# Description: Main file used for running the classifier on the board
# Date: 2023-05-10
#==============================================================================


import numpy as np
#import tensorflow as tf
import os
import serial
from segmentation.segmentation import data_segmentation
from feature.feature import fe
from dimension.dimension import pca_func
from sklearn.neighbors import KNeighborsClassifier
import pickle


#==============================================================================
# Function and class for reading data
ser = serial.Serial("/dev/ttyACM1")
class ReadLine:
    def __init__(self, s):
        serlf.buf = bytearray()
        self.s = s
    def readline(self):
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[:i+1]
            self.buf = self.buf[i+1:]
            return r
        while True:
            i = max(1, min(2048, self.s.sin_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buf + data[:i+1]
                self.buf[0:] = data[i+1:]
                return r
            else:
                self.buf.extend(data)
def load_data():
    #r1.ReadLine(ser)
    sample_counter = 0
    buf = []
    number_samples_to_load = 375
    while sample_counter < number_samples_to_load:
        read_voltage_from_adc = ser.readline()
        read_voltage_from_adc = read_voltage_from_adc.decode('utf-8').rstrip('\n').rstrip('\r')
        print(read_voltage_from_adc)
        buf.append(read_voltage_from_adc)
        sample_counter += 1
    return buf
#==============================================================================
# Loads the ML model or data
def load_model(file_name):
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
def main():
    file_name = "./trained_scikit_models/trained_knn_classifier.txt"
    sampling_frequency = 1400
    window_size = 0.25
    overlap = 0.125
    number_classes = 11
    number_principal_components = 5

    # Load the model from file
    model = load_model(file_name)

    while(True):
        # Load data
        data = load_data()

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
    main()
#==============================================================================

