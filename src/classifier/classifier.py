#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Trains classifier
# Date: 2023-04-25
#==============================================================================


import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
# XGBoost doesn't work for python 32-bit
from xgboost import XGBClassifier
# TensorFlow doesn't work for python 3.7
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.data import Dataset
from optimizer.optimizer import ga

import config


#==============================================================================
# Converts from class vector to integer class
# Scikit doesn't accept vector classes, it expects integers
def class_vector_to_integer(labels):
    # Temporary numpy copy of the data frame labels
    temp_labels = labels.to_numpy(copy=True)
    # New array that will hold integer for which class it belongs to instead of class vector
    int_arr = []

    # Go through all segments
    for vector in temp_labels:
        # Convert vector to int
        class_int = np.argmax(vector)
        # Add int to the new int array
        int_arr.append(class_int)

    # Return int array that will replace class vector array
    return pd.DataFrame(np.array(int_arr), columns = ['Integer labels'])
#==============================================================================
# Converts from integer classes to class vector
def integer_to_class_vector(labels, num_classes):
    # Temporary numpy copy of the data frame labels
    temp_labels = labels.to_numpy(copy=True)
    # New array that will hold all class vectors
    vector_arr = []

    # Go through all segments
    for int_class in temp_labels:
        # Create array of zeros
        class_vector = np.zeros(num_classes)
        # Set the position (class) which it belongs to to 1
        class_vector[int_class] = 1
        vector_arr.append(class_vector)

    # Return array containing class vectors
    return pd.DataFrame(np.array(vector_arr), columns = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9', 'Class 10'])
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

        data_train, data_test = data.iloc[train], data.iloc[test]
        label_train, label_test = labels.iloc[train], labels.iloc[test]

        # Fix class imbalance in training data
        data_train, label_train = class_imb(data_train, label_train)

        # Create LDA model
        clf = PMMLPipeline([("classifier", LinearDiscriminantAnalysis(n_components=num_components, solver=lda_solver))])

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
    accuracy = accuracy_score(labels, predictions)
    print(accuracy)
    print(confusion_matrix(labels, predictions))

    # Save classifier to pmml file
    sklearn2pmml(clf, "./trained_scikit_models/lda.pmml", with_repr = True)
    print('Saved LDA classifier to:', "./trained_scikit_models/lda.pmml")
    print("=======================================================================")
#==============================================================================
# Train SVM
def support_vector_machine(data, labels, num_splits, kernel, gamma, decision_function_shape):
    # Scikit doesn't accept vector classes, it expects integers
    labels = class_vector_to_integer(labels)

    total_accuracy = 0
    # Performs k fold cross validation
    kfold = KFold(n_splits=num_splits, shuffle=True)
    for train, test in kfold.split(data):

        data_train, data_test = data.iloc[train], data.iloc[test]
        label_train, label_test = labels.iloc[train], labels.iloc[test]

        # Fix class imbalance in training data
        data_train, label_train = class_imb(data_train, label_train)

        # Create SVM classifier
        clf = PMMLPipeline([("classifier", make_pipeline(StandardScaler(), SVC(kernel=kernel, gamma=gamma, decision_function_shape=decision_function_shape)))])

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
    accuracy = accuracy_score(labels, predictions)
    print(accuracy)
    print(confusion_matrix(labels, predictions))

    # Save classifier to pmml file
    sklearn2pmml(clf, "./trained_scikit_models/svm.pmml", with_repr = True)
    print('Saved SVM classifier to:', "./trained_scikit_models/svm.pmml")
    print("=======================================================================")
#==============================================================================
# Train KNN
def knn(data, labels, num_splits, num_neighbors, weight_function, leaf_size):
    # Scikit doesn't accept vector classes, it expects integers
    labels = class_vector_to_integer(labels)

    total_accuracy = 0
    # Performs k fold cross validation
    kfold = KFold(n_splits=num_splits, shuffle=True)
    for train, test in kfold.split(data):

        data_train, data_test = data.iloc[train], data.iloc[test]
        label_train, label_test = labels.iloc[train], labels.iloc[test]

        # Fix class imbalance in training data
        data_train, label_train = class_imb(data_train, label_train)

        # Create KNN classifier
        clf = PMMLPipeline([("classifier", KNeighborsClassifier(n_neighbors = num_neighbors, weights=weight_function, leaf_size=leaf_size))])

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
    accuracy = accuracy_score(labels, predictions)
    print(accuracy)
    print(confusion_matrix(labels, predictions))

    # Save classifier to pmml file
    sklearn2pmml(clf, "./trained_scikit_models/knn.pmml", with_repr = True)
    print('Saved KNN classifier to:', "./trained_scikit_models/knn.pmml")
    print("=======================================================================")
#==============================================================================
# Train MLP
def mlp(data, labels, layers, activation_func, solver_func, learning_rate_model, alpha, iterations, num_splits, train_set_proportion):
    # Scikit doesn't accept vector classes, it expects integers
    labels = class_vector_to_integer(labels)

    total_accuracy = 0
    # Performs k fold cross validation
    kfold = KFold(n_splits=num_splits, shuffle=True)
    for train, test in kfold.split(data):

        data_train, data_test = data.iloc[train], data.iloc[test]
        label_train, label_test = labels.iloc[train], labels.iloc[test]

        # Fix class imbalance in training data
        data_train, label_train = class_imb(data_train, label_train)

        # Create MLP model
        clf = PMMLPipeline([("classifier", MLPClassifier(hidden_layer_sizes=(layers,), activation=activation_func, solver=solver_func, learning_rate=learning_rate_model,
            learning_rate_init = alpha, max_iter=iterations, early_stopping=False, validation_fraction=1-train_set_proportion))])

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
    accuracy = accuracy_score(labels, predictions)
    print(accuracy)
    print(confusion_matrix(labels, predictions))

    # Save classifier to pmml file
    sklearn2pmml(clf, "./trained_scikit_models/mlp.pmml", with_repr = True)
    print('Saved MLP classifier to:', "./trained_scikit_models/mlp.pmml")
    print("=======================================================================")
#==============================================================================
# XGBoost doesn't work for python 32-bit
#==============================================================================
# XGBoost classifier
def xgboost_classifier(data, labels, train_set_proportion, num_splits, num_classes):
    # Scikit doesn't accept vector classes, it expects integers
    labels = class_vector_to_integer(labels)

    total_accuracy = 0
    # Performs k fold cross validation
    kfold = KFold(n_splits=num_splits, shuffle=True)
    for train, test in kfold.split(data):

        data_train, data_test = data.iloc[train], data.iloc[test]
        label_train, label_test = labels.iloc[train], labels.iloc[test]

        # Fix class imbalance in training data
        data_train, label_train = class_imb(data_train, label_train)

        # Create multi class XGBoost classifier
        clf = PMMLPipeline([("classifier", XGBClassifier(objective='multi:softmax', num_class = num_classes, verbosity = 0))])

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
    accuracy = accuracy_score(labels, predictions)
    print(accuracy)
    print(confusion_matrix(labels, predictions))

    # Save classifier to pmml file
    sklearn2pmml(clf, "./trained_scikit_models/xgb.pmml", with_repr = True)
    print('Saved XGBoost classifier to:', "./trained_scikit_models/xgb.pmml")
    print("=======================================================================")
#==============================================================================
#==============================================================================
# TensorFlow doesn't work for python 3.7
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

        data_train, data_test = data.iloc[train], data.iloc[test]
        label_train, label_test = labels.iloc[train], labels.iloc[test]

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
    accuracy = accuracy_score(prediction_labels, predictions)
    print(accuracy)
    print(confusion_matrix(prediction_labels, predictions))

    # Return the classifier
    return model
#==============================================================================
