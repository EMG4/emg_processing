#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Performs feature extraction
# Date: 2023-04-25
#==============================================================================


import tsfel
import numpy as np
import pandas as pd
import pprint
import feature.feature_list as fl


#==============================================================================
# Feautre extraction
def fe(segment_arr, sampling_frequency):
    # New array that will contain processed segment_arr, since we cannot directly change in segment_arr
    a = []

    # Retrieve feature configuration file
    cfg = fl.feature_list
    # Get features by domain
    #cfg = tsfel.get_features_by_domain("temporal")

    # Perform feature extraction on every segment
    for item in segment_arr:
        # Have to convert to pandas since tsfel doesn't work on numpy, then transform back
        temp_panda = pd.DataFrame(np.hstack(item), columns=["EMG signal"])
        # Perform feature extraction
        temp_result = tsfel.time_series_features_extractor(cfg, temp_panda, fs = sampling_frequency, verbose = 0)
        
        # Transform back to numpy
        temp_numpy = temp_result.to_numpy(copy=True)
        a.append(temp_numpy[0])


    # Return numpy array containing the feature extracted segments
    return np.array(a)
#==============================================================================
