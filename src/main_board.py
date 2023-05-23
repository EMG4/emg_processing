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
from filtering.filter import bandpass, notch
from segmentation.segmentation import data_segmentation
from feature.feature import fe
from dimension.dimension import pca_func
from sklearn.neighbors import KNeighborsClassifier
from jpmml_evaluator import make_evaluator


#==============================================================================
# Function and class for reading data
# Class for reading data
ser = serial.Serial("/dev/ttyACM1")
class ReadLine:
    def __init__(self, s):
        self.buf = bytearray()
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
#==============================================================================
# Function for reading data
def load_data():
    read_voltage_from_adc = ReadLine(ser)
    sample_counter = 0
    buf = []
    number_samples_to_load = 375
    while sample_counter < number_samples_to_load:
        read_voltage_from_adc = ser.readline()
        read_voltage_from_adc = read_voltage_from_adc.decode('utf-8').rstrip('\n').rstrip('\r')
        #print(read_voltage_from_adc)
        buf.append(int(read_voltage_from_adc))
        sample_counter += 1
    return buf
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
    file_name = "./trained_scikit_models/knn.pmml"
    sampling_frequency = 1400
    window_size = 0.25
    overlap = 0.125
    number_classes = 11
    number_principal_components = 2

    # Load the model from file
    model = load_model(file_name)

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
#==============================================================================
if __name__ == "__main__":
    main()
#==============================================================================

