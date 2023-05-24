#! /usr/bin/python3
#==============================================================================
# Authors: Carl Larsson, Pontus Svensson, Samuel Wågbrant
# Description: Main file used for running the classifier on the board
# Date: 2023-05-10
#==============================================================================


from collection.data_collection import load_data
import os
import numpy as np
import keyboard as kb
import threading
from collection.data_collection import load_data
from filtering.filter import bandpass, notch
from segmentation.segmentation import data_segmentation
from feature.feature import fe
from dimension.dimension import pca_func
from sklearn.neighbors import KNeighborsClassifier
from jpmml_evaluator import make_evaluator


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
		elif key == 'd':
			label = 10
		else:
			label = 0


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
#=============================================================
def main():
    file_name = "./trained_scikit_models/knn.pmml"
    sampling_frequency = 1000
    window_size = 0.25
    overlap = 0.125
    number_classes = 11
    number_principal_components = 2
    t1 = threading.Thread(target=readKey)
    t1.start()
    header = ["Integer labels", "key"]

    # Load the model from file
    model = load_model(file_name)

    input("Press enter to start classification...")
    with open('dataset.txt', 'w') as f:

        while True:
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
            #print(prediction[['Integer labels']])
            prediction['key'] = label
            toPrint = prediction[header].to_string(header=False, index=False, columns=header)
            # print(str(prediction[['Integer labels']]))
            #print(toPrint)
            f.write(toPrint+'\n')
            f.flush()
            # f.write(toPrint + '\t' + str(label)+'\n')

    f.close()

#==============================================================================
if __name__ == "__main__":
    main()
#==============================================================================

