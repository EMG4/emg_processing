#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Performs dimensionallity reduction on input file and provides output file
# Date: 2023-04-25
#==============================================================================


import pandas as pd
from feature import fe
from dimension import pca_func, ofnda_func
from classifier import lda, mlp, cnn


#==============================================================================
# Loads the data
def load_data(file_name):
    # Load data
    data = pd.read_csv(file_name, sep='\t')

    return data
#==============================================================================
# Main function
def main():
    pca_number_components = 7 
    lda_number_components = 10

    # Load data (csv)
    csv_data = load_data("data.txt")

    # First column is the data
    raw_data = csv_data.iloc[:, :1]
    # Second column is labels
    labels = csv_data.iloc[:, :2]
    

    # Performs feature extraction
    feature_data = fe(raw_data)


    # Chooses dimension reduction
    if(pca)
        data = pca_func(feature_data, pca_number_components)
    elif(ofnda)
        data = ofnda_func()
    else
        data = feature_data


    # Chooses classifier
    if(lda)
        classifier = lda(data, labels, )

    elif(mlp)
        classifier = mlp(data, labels )

    elif(ann)
        classifier = ann() 

    elif(cnn)
        classifier = cnn() 
    else
        print("no classifier is chosen")
        exit()


    # Saves the trained classifier to a json file


#==============================================================================
if __name__ == "__main__"":
    main()
