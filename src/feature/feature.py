#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Performs feature extraction
# Date: 2023-04-25
#==============================================================================


import tsfel
import numpy as np
import pandas as pd
import feature.feature_list as fl


#==============================================================================
# Feautre extraction
def fe(segment_arr, sampling_frequency):
    # Retrieve feature configuration file
    cfg = fl.feature_list
    # Get features by domain
    #cfg = tsfel.get_features_by_domain("temporal")

    # Perform feature extraction
    temp_segment_arr = tsfel.time_series_features_extractor(cfg, segment_arr, fs = sampling_frequency, verbose = 0)

    # Return pandas dataframe containing the feature extracted segments
    return temp_segment_arr
#==============================================================================
