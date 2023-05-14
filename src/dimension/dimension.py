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
    # Can not have more components than min of features and samples
    min_feature_samples = np.min(segment_arr.shape)
    if(min_feature_samples<numb_components):
        print(f"Number of principal components were reduced from", {numb_components}, "to", {min_feature_samples},", because of min(features, samples)")
        numb_components = min_feature_samples

    # Run PCA on entire segment_arr
    # Standardize data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(segment_arr)
    
    # Perform PCA
    pca = PCA(n_components = numb_components)
    pca.fit(standardized_data)

    # Return PCA processed segment array
    return pca.fit_transform(standardized_data)
#==============================================================================
# Function perfoming OFNDA
def ofnda_func():
    pass
#==============================================================================
