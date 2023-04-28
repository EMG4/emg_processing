#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Trains classifier
# Date: 2023-04-25
#==============================================================================


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


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

    # Fix class imbalance in training set
    training_data, training_data_labels = class_imb(training_data, training_data_labels)

    # Create LDA model
    clf = LinearDiscriminantAnalysis(n_components=num_components)
    # Train on test data
    clf.fit(training_data, training_data_labels)


    # Check accuracy
    clf.score(validation_data, validation_data_labels)

    return clf
#==============================================================================
# Train MLP
def mlp(data, labels, train_set_proportion, layers, activation_func, solver_func, learning_rate_model, alpha, iterations):
    # Split into training set and validation set
    training_data, validation_data, training_data_labels, validation_data_labels = train_test_split(data, labels, train_size=train_set_proportion)

    # Fix class imbalance in training set
    training_data, training_data_labels = class_imb(training_data, training_data_labels)

    # Create MLP model
    clf = MLPClassifier(hidden_layer_sizes=(layers,), activation=activation_func, solver=solver_func, learning_rate=learning_rate_model, learning_rate_init = alpha, max_iter=iterations)
    # Train on test data
    clf.fit(training_data, training_data_labels)
    

    # Check accuracy
    clf.score(validation_data, validation_data_labels)

    return clf
#==============================================================================
# Train ANN
def ann(data, labels, num_splits, dropout_rate, input_dim, layers, alpha, num_epochs, activation_func, neurons, max_norm):
    # Fix class imbalance
    data, labels = class_imb(data, labels)

    total_accuracy = 0.0
    # Performs k fold cross validation
    kfold = KFold(num_splits, True)
    for train, test in kfold.split(data):

        data_train, data_test = data[train], data[test]
        label_train, label_test = labels[train], labels[test]

        # Sequential NN model
        model = Sequential()
        # Add drop out to prevent overfitting
        # Create input layer
        model.add(Dropout(dropout_rate, input_shape=(input_dim,)))
        # Creates hidden layers
        for l in layers:
            model.add(Dense(neurons, activation=activation_func, kernel_constraint=MaxNorm(max_norm)))
            model.add(Dropout(dropout_rate))
        # Output layer
        model.add(Dense(1, activation='sigmoid'))

        # Define loss function. optimizer/solver, and any other metrics to report
        optimiz = Adam(learning_rate=alpha)
        model.compile(loss='categorical_crossentropy', optimizer=optimiz, metrics=['accuracy'])

        # Train the model
        model.fit(data_train, label_train, epochs=num_epochs)

        # Evalute the model
        _, accuracy = model.evaluate(data_test, label_test)
        total_accuracy = total_accuracy + accuracy

    # Calculate and print mean accuracy
    total_accuracy = total_accuracy/num_splits
    print('Accuracy: %.2f' % (total_accuracy*100))

    return model
#==============================================================================
# Train CNN
def cnn():
    pass
#==============================================================================
