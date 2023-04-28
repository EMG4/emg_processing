#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Performs dimensionallity reduction on input file and provides output file
# Date: 2023-04-25
#==============================================================================


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#==============================================================================
# Function peforming PCA
def pca_func(segment_arr, numb_components = 2):
    
    for item in segment_arr:
        print(item)
        print(item.shape)
        # Standardize data
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(item)

        # Perform PCA
        pca = PCA(n_components = numb_components)
        pca.fit(standardized_data)

        item = pca.fit_transform(standardized_data)
        print(item)
        print(item.shape)


    return segment_arr
#==============================================================================
# Function perfoming OFNDA
def ofnda_func():
    pass
#==============================================================================
