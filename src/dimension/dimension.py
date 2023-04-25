#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Performs dimensionallity reduction on input file and provides output file
# Date: 2023-04-25
#==============================================================================

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#==============================================================================
# Function peforming PCA
def pca_func(data, numb_components = 2):
    
    # Standardize data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data[:, 1:])

    # Perform PCA
    pca = PCA(n_components = numb_components)
    pca.fit(standardized_data)

    projected = pca.fit_transform(standardized_data)


    return projected_data
#==============================================================================
# Function perfoming OFNDA
#==============================================================================
