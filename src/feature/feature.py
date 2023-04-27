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
    domain = ("statistical", "temporal")
    cfg = tsfel.get_feature_by_domain(domain, None)
    
    # Perform feature extraction
    return tsfel.time_series_featues_extractor(cfg, segment_arr)
#==============================================================================
