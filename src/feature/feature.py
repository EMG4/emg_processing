#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Performs feature extraction on input file and provides output file
# Date: 2023-04-25
#==============================================================================


import tsfel
import numpy as np
import pandas as pd


#==============================================================================
# Feautre extraction
def fe(segment_arr, sampling_frequency):

    a = []

    # Retrieve feature configuration file
    cfg = tsfel.get_features_by_domain(domain = None, json_path=None)
    
    # Perform feature extraction on every segment
    for item in segment_arr:
        # Have to convert to pandas since tsfel doesn't work on numpy, then transform back
        temp_panda = pd.DataFrame(np.hstack(item), columns=["EMG signal"])
        temp_result = tsfel.time_series_features_extractor(cfg, temp_panda, fs = sampling_frequency, verbose = 0)
        
        temp_numpy = temp_result.to_numpy(copy=True)
        a.append(temp_numpy[0])


    # Perform feature extraction
    temp_arr = np.array(a)
    return temp_arr
#==============================================================================
