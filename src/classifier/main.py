#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Trains classifier
# Date: 2023-04-25
#==============================================================================

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler



#==============================================================================
# Loads the data
def load_data(file_name):
    data = np.loadtxt(file_name, delimiter = " ")

    return data
#==============================================================================
# Fix class imbalance
def class_imb(training_data, training_data_label):
    ros = RandomOverSampler()
    training_data_res, training_data_label_res = ros.fit_resample(training_data, training_data_label)

    return training_data_res, training_data_label_res
#==============================================================================
# LDA
def lda(training_data, training_data_label, validation_data, validation_data_label):
    # Create LDA model
    clf = LinearDiscriminantAnalysis(n_components=None)
    # Train on test data
    clf.fit(training_data, training_data_label)


    # Check accuracy
    clf.score(validation_data, validation_data_label)
#==============================================================================
# ANN/MLP
def mlp(training_data, training_data_label, validation_data, validation_data_label):
    # Create MLP model
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', learning_rate='constant', max_iter=200)
    # Train on test data
    clf.fit(training_data, training_data_label)
    

    # Check accuracy
    clf.score(validation_data, validation_data_label)
#==============================================================================
# CNN
def cnn():

#==============================================================================
# Main function
def main():
    data = load_data("data.txt")
    
    # Fix class imbalance
    training_data, training_data_label = class_imb(training_data, training_data_label)


#==============================================================================
if __name__ == "__main__"":
    main()
