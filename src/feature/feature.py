#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Performs feature extraction on input file and provides output file
# Date: 2023-04-25
#==============================================================================


import tsfel


#==============================================================================
# Feautre extraction
def fe(segment_arr):

    # Retrieve feature configuration file to extract temporal and statiscial time domain features
    cfg = tsfel.get_features_by_domain(domain = None, json_path=None)
    
    for item in segment_arr:
        item = tsfel.time_series_featues_extractor(cfg, item)

    # Perform feature extraction
    return segment_arr
#==============================================================================
