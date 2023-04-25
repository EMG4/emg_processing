#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Performs feature extraction on input file and provides output file
# Date: 2023-04-25
#==============================================================================

import pandas as pd
import tsfel


#==============================================================================
# Loads the data
def load_data(file_name):
    # Load data
    data = pd.read_csv(file_name)

    return data
#==============================================================================
# Feautre extraction
def fe(data):
    # Retrieve feature configuration file to extract temporal and statiscial time domain features
    domain = ("statistical", "temporal")
    cfg = tsfel.get_feature_by_domain(domain, None)

    return tsfel.time_series_featues_extractor(cfg, data)
#==============================================================================
# Main function
def main():
    data = load_data("data.txt")

    features = fe(data)

#==============================================================================
if __name__ == "__main__"":
    main()
