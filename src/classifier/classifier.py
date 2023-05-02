#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Trains classifier
# Date: 2023-04-25
#==============================================================================


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset


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

    return np.array(vector_arr)
#==============================================================================
# Fix class imbalance
def class_imb(training_data, training_data_labels):
    ros = RandomOverSampler()
    training_data_res, training_data_labels_res = ros.fit_resample(training_data, training_data_labels)

    return training_data_res, training_data_labels_res
#==============================================================================
# Train LDA
def lda(data, labels, train_set_proportion, num_components):
    # Split into training set and validation set
    training_data, validation_data, training_data_labels, validation_data_labels = train_test_split(data, labels, train_size=train_set_proportion)

    # Scikit doesn't accept vector classes, it expects integers
    training_data_labels = class_vector_to_integer(training_data_labels)
    validation_data_labels = class_vector_to_integer(validation_data_labels)

    # Fix class imbalance in training set
    training_data, training_data_labels = class_imb(training_data, training_data_labels)

    # Create LDA model
    clf = LinearDiscriminantAnalysis(n_components=num_components)
    # Train on test data
    clf.fit(training_data, training_data_labels)


    # Check accuracy
    print(clf.score(validation_data, validation_data_labels))
    predictions = clf.predict(validation_data)
    print(confusion_matrix(validation_data_labels, predictions))

    return clf
#==============================================================================
# Train MLP
def mlp(data, labels, train_set_proportion, layers, activation_func, solver_func, learning_rate_model, alpha, iterations):
    # Split into training set and validation set
    training_data, validation_data, training_data_labels, validation_data_labels = train_test_split(data, labels, train_size=train_set_proportion)
    # Scikit doesn't accept vector classes, it expects integers
    training_data_labels = class_vector_to_integer(training_data_labels)
    validation_data_labels = class_vector_to_integer(validation_data_labels)

    # Fix class imbalance in training set
    training_data, training_data_labels = class_imb(training_data, training_data_labels)

    # Create MLP model
    clf = MLPClassifier(hidden_layer_sizes=(layers,), activation=activation_func, solver=solver_func, learning_rate=learning_rate_model, learning_rate_init = alpha, max_iter=iterations)
    # Train on test data
    clf.fit(training_data, training_data_labels)
    

    # Check accuracy
    print(clf.score(validation_data, validation_data_labels))
    predictions = clf.predict(validation_data)
    print(confusion_matrix(validation_data_labels, predictions))

    return clf
#==============================================================================
# Train ANN
def ann(data, labels, num_splits, dropout_rate, input_dim, layers, alpha, num_epochs, activation_func, neurons, max_norm, b_size, num_classes):

    # Batch size must be smaller than number of samples
    number_samples = data.shape[0]
    if(number_samples<b_size):
        print(f"Batch size was reduced from", {b_size}, "to", {number_samples},", because of number of samples")
        b_size = number_samples

    # Class_imb expects integer classes, not class vector
    # Fix class imbalance
    data, labels = class_imb(data, class_vector_to_integer(labels))
    # Convert back from integer classes to class vector since ANN keras expects class vector
    labels = integer_to_class_vector(labels, num_classes)

    total_accuracy = 0.0
    # Performs k fold cross validation
    kfold = KFold(n_splits = num_splits)
    for train, test in kfold.split(data):

        data_train, data_test = data[train], data[test]
        label_train, label_test = labels[train], labels[test]

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
            model.add(Dense(neurons, activation=activation_func, kernel_constraint=MaxNorm(max_norm)))
            model.add(Dropout(dropout_rate))
        # Output layer
        model.add(Dense(num_classes, activation='sigmoid'))

        # Define loss function. optimizer/solver, and any other metrics to report
        optimiz = Adam(learning_rate=alpha, name="Adam")
        model.compile(loss='categorical_crossentropy', optimizer=optimiz, metrics=['accuracy'])

        # Train the model
        model.fit(train_dataset, epochs=num_epochs, validation_data=validation_dataset, verbose=0, batch_size = b_size)

        # Evalute the model
        _, accuracy = model.evaluate(data_test, label_test)
        print('Accuracy: %.2f' % (accuracy*100))
        total_accuracy = total_accuracy + accuracy

        # Confusion matrix
        #predictions = model.predict(data_test)
        #prediction_labels = class_vector_to_integer(label_test)
        #print(confusion_matrix(prediction_labels, predictions))

    # Calculate and print mean accuracy
    total_accuracy = total_accuracy/num_splits
    print('Total Accuracy: %.2f' % (total_accuracy*100))

    return model
#==============================================================================
# Train CNN
def cnn():
    pass
#==============================================================================
