#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Trains classifier
# Date: 2023-04-25
#==============================================================================


import numpy as np
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.data import Dataset
from xgboost import XGBClassifier

from optimizer.optimizer import ga
import config


#==============================================================================
# Converts from class vector to integer class
# Scikit doesn't accept vector classes, it expects integers
def class_vector_to_integer(labels):
    # New array that will hold integer for which class it belongs to instead of class vector
    int_arr = []
    # Go through all segments
    for vector in labels:
        # Convert vector to int
        class_int = np.argmax(vector)
        # Add int to the new int array
        int_arr.append(class_int)
    # Return int array that will replace class vector array
    return np.array(int_arr)
#==============================================================================
# Converts from integer classes to class vector
def integer_to_class_vector(labels, num_classes):
    # New array that will hold all class vectors
    vector_arr = []

    # Go through all segments
    for int_class in labels:
        # Create array of zeros
        class_vector = np.zeros(num_classes)
        # Set the position (class) which it belongs to to 1
        class_vector[int_class] = 1
        vector_arr.append(class_vector)

    # Return array containing class vectors
    return np.array(vector_arr)
#==============================================================================
# Fix class imbalance
def class_imb(data, labels):
    # Using random over smampling
    ros = RandomOverSampler()
    data_res, labels_res = ros.fit_resample(data, labels)

    # Return new resampled data and labels
    return data_res, labels_res
#==============================================================================
# Train LDA
def lda(data, labels, num_components, num_splits, lda_solver):
    # Scikit doesn't accept vector classes, it expects integers
    labels = class_vector_to_integer(labels)

    total_accuracy = 0
    # Performs k fold cross validation
    kfold = KFold(n_splits=num_splits, shuffle=True)
    for train, test in kfold.split(data):

        data_train, data_test = data[train], data[test]
        label_train, label_test = labels[train], labels[test]

        # Fix class imbalance in training data
        data_train, label_train = class_imb(data_train, label_train)

        # Create LDA model
        clf = LinearDiscriminantAnalysis(n_components=num_components, solver=lda_solver)

        # Train model
        clf.fit(data_train, label_train)

        # Evalute the model
        accuracy = clf.score(data_test, label_test)
        total_accuracy = total_accuracy + accuracy


    # Calculate and print mean accuracy
    total_accuracy = total_accuracy/num_splits
    print('Total Accuracy: %.2f' % (total_accuracy*100))

    # Print confusion matrix
    predictions = clf.predict(data)
    print(confusion_matrix(labels, predictions))

    # Return the classifier
    return clf
#==============================================================================
# Train MLP
def mlp(data, labels, layers, activation_func, solver_func, learning_rate_model, alpha, iterations, num_splits, train_set_proportion):
    # Scikit doesn't accept vector classes, it expects integers
    labels = class_vector_to_integer(labels)

    total_accuracy = 0
    # Performs k fold cross validation
    kfold = KFold(n_splits=num_splits, shuffle=True)
    for train, test in kfold.split(data):

        data_train, data_test = data[train], data[test]
        label_train, label_test = labels[train], labels[test]

        # Fix class imbalance in training data
        data_train, label_train = class_imb(data_train, label_train)

        # Create MLP model
        clf = MLPClassifier(hidden_layer_sizes=(layers,), activation=activation_func, solver=solver_func, learning_rate=learning_rate_model, learning_rate_init = alpha, max_iter=iterations, early_stopping=False, validation_fraction=1-train_set_proportion)

        # Train model
        clf.fit(data_train, label_train)

        # Evalute the model
        accuracy = clf.score(data_test, label_test)
        total_accuracy = total_accuracy + accuracy


    # Calculate and print mean accuracy
    total_accuracy = total_accuracy/num_splits
    print('Total Accuracy: %.2f' % (total_accuracy*100))

    # Print confusion matrix
    predictions = clf.predict(data)
    print(confusion_matrix(labels, predictions))

    # Return the classifier
    return clf
#==============================================================================
# Train ANN
def ann(data, labels, num_splits, dropout_rate, input_dim, layers, solver_func, num_epochs, activation_func, neurons, b_size, num_classes, num_solutions, num_generations, num_parents_mating):

    # Batch size must be smaller than number of samples
    number_samples = data.shape[0]
    if(number_samples<b_size):
        print(f"Batch size was reduced from", {b_size}, "to", {number_samples},", because of number of samples")
        b_size = number_samples


    # Run GA to optimize parameters
    # Need to create the structure of the ANN
    # Sequential NN model
    config.model = Sequential()
    # Add drop out to prevent overfitting
    # Create input layer
    config.model.add(Dropout(dropout_rate, input_shape=(input_dim,)))
    # Creates hidden layers
    for l in range(layers):
        config.model.add(Dense(neurons, activation=activation_func))
        config.model.add(Dropout(dropout_rate))
    # Output layer
    config.model.add(Dense(num_classes, activation='softmax'))

    # Run GA to find optimal parameters
    # Need to use global variables since fitness function doesn't accept arguments
    config.ga_data = data
    config.ga_labels = labels
    best_solution_weights = ga(num_solutions, num_generations, num_parents_mating)


    # Used to get overall accuracy
    total_accuracy = 0.0
    # Performs k fold cross validation
    kfold = KFold(n_splits = num_splits, shuffle=True)
    for train, test in kfold.split(data):

        data_train, data_test = data[train], data[test]
        label_train, label_test = labels[train], labels[test]

        # Class_imb expects integer classes, not class vector
        # Fix class imbalance
        data_train, label_train = class_imb(data_train, class_vector_to_integer(label_train))
        # Convert back from integer classes to class vector since ANN keras expects class vector
        label_train = integer_to_class_vector(label_train, num_classes)

        train_dataset = Dataset.from_tensor_slices((data_train, label_train))
        validation_dataset = Dataset.from_tensor_slices((data_test, label_test))
        train_dataset = train_dataset.batch(b_size, drop_remainder=False)
        validation_dataset = validation_dataset.batch(b_size, drop_remainder=False)


        # Sequential NN model
        model = Sequential()
        # Add drop out to prevent overfitting
        # Create input layer
        model.add(Dropout(dropout_rate, input_shape=(input_dim,)))
        # Creates hidden layers
        for l in range(layers):
            model.add(Dense(neurons, activation=activation_func))
            model.add(Dropout(dropout_rate))
        # Output layer
        # One neuron for each class, sigmoid so we get how certain the classifier is on the data belonging to each class
        model.add(Dense(num_classes, activation='softmax'))

        # Set the parameters found to be optimal by the GA
        model.set_weights(best_solution_weights)

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=solver_func, metrics=['accuracy'])


        # Evalute the model
        _, accuracy = model.evaluate(data_test, label_test, verbose = 0)
        print('Accuracy: %.2f' % (accuracy*100))
        total_accuracy = total_accuracy + accuracy


    print(model.summary())

    # Calculate and print mean accuracy
    total_accuracy = total_accuracy/num_splits
    print('Total Accuracy: %.2f' % (total_accuracy*100))

    # Confusion matrix
    predictions = class_vector_to_integer(model.predict(data, verbose = 0))
    prediction_labels = class_vector_to_integer(labels)
    print(confusion_matrix(prediction_labels, predictions))

    # Return the classifier
    return model
#==============================================================================
# XGBoost classifier
def xgboost_classifier(data, labels, train_set_proportion, num_splits, num_classes):
    run_kfold = 0

    if(run_kfold):
        # Scikit doesn't accept vector classes, it expects integers
        labels = class_vector_to_integer(labels)

        total_accuracy = 0
        # Performs k fold cross validation
        kfold = KFold(n_splits=num_splits, shuffle=True)
        for train, test in kfold.split(data):

            data_train, data_test = data[train], data[test]
            label_train, label_test = labels[train], labels[test]

            # Fix class imbalance in training data
            data_train, label_train = class_imb(data_train, label_train)

            # Create multi class XGBoost classifier
            bst = XGBClassifier(objective='multi:softmax', num_class = num_classes, verbosity = 0)

            # Train model
            bst.fit(data_train, label_train)

            # Evalute the model
            accuracy = bst.score(data_test, label_test)
            total_accuracy = total_accuracy + accuracy


        # Calculate and print mean accuracy
        total_accuracy = total_accuracy/num_splits
        print('Total Accuracy: %.2f' % (total_accuracy*100))

        # Print confusion matrix
        predictions = bst.predict(data)
        print(confusion_matrix(labels, predictions))

    else:
        # Scikit doesn't accept vector classes, it expects integers
        labels = class_vector_to_integer(labels)
        # Split into training set and validation set
        training_data, validation_data, training_data_labels, validation_data_labels = train_test_split(data, labels, train_size=train_set_proportion)

        # Fix class imbalance in training set
        training_data, training_data_labels = class_imb(training_data, training_data_labels)


        # Create multi class XGBoost classifier
        bst = XGBClassifier(objective='multi:softmax', num_class = num_classes, verbosity = 0)

        # Train classifer
        bst.fit(training_data, training_data_labels)


        # Evalute classifier on validation set
        pred = bst.predict(validation_data)
        accuracy = accuracy_score(validation_data_labels, pred)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        # Print confusion matrix
        predictions = bst.predict(data)
        print(confusion_matrix(labels, predictions))

    # Return the tree
    return bst
#==============================================================================
