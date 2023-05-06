#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Performs dimensionallity reduction
# Date: 2023-04-25
#==============================================================================


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


#==============================================================================
# Function peforming PCA
def pca_func(segment_arr, numb_components):
    run_entire = 1
    
    # Can not have more components than min of features and samples
    min_feature_samples = np.min(segment_arr.shape)
    if(min_feature_samples<numb_components):
        print(f"Number of principal components were reduced from", {numb_components}, "to", {min_feature_samples},", because of min(features, samples)")
        numb_components = min_feature_samples

    # I think this approach is correct
    # Run PCA on entire segment_arr
    if(run_entire == 1):
        # Standardize data
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(segment_arr)
    
        # Perform PCA
        pca = PCA(n_components = numb_components)
        pca.fit(standardized_data)

        # Return PCA processed segment array
        return pca.fit_transform(standardized_data)
    
    # Run PCA on each individual segment
    else:
        # Temp array since we cannot change in segment_arr
        a = []

        # Do PCA on every segment
        for item in segment_arr:
            # Need to give transposed, otherwise it divides by 0
            temp_item = np.array([item]).transpose()
            # Standardize data
            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(temp_item)

            # Perform PCA
            pca = PCA(n_components = numb_components)
            pca.fit(standardized_data)

            temp = pca.fit_transform(standardized_data)
            # Transpose back
            temp = temp.transpose()[0]

            a.append(temp)


        # Return PCA processed segment array
        return np.array(a)
#==============================================================================
# Function perfoming OFNDA
def ofnda_func():
    pass
#==============================================================================
