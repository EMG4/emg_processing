#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Performs segmentation
# Date: 2023-04-27
#==============================================================================


import pyemgpipeline as pep
import numpy as np


#==============================================================================
# Performs segmentation
def segmentation(data, labels, sampling_rate, window_size, overlap, num_classes):
    # Needed for compilation incase for loop isn't executed
    segment_arr = None
    label_arr = None

    # Find length of data
    data_length = data.shape[0]
    t = 0
    # Segment data with overlapping windows
    while(True):
        data_segment = pep.wrappers.EMGMeasurement(data, hz = sampling_rate)
        label_segment = pep.wrappers.EMGMeasurement(labels, hz = sampling_rate)
        data_segment.apply_segmenter(beg_ts = 0+t, end_ts = window_size+t)
        label_segment.apply_segmenter(beg_ts = 0+t, end_ts = window_size+t)
        
        # Create array first time we run
        if t == 0:
            # Array brackets to correctly store each segment as an array withing the larger array
            segment_arr = np.array([data_segment.data])

            # Figure out majoirty class of that segment
            # Create vector of zeros, majority class position is set to 1 since this segment belongs to said class
            # Add class vector to array of labels corresponding to each segment
            majority_class = round(np.average(label_segment.data))
            class_vector = np.zeros(num_classes)
            class_vector[majority_class] = 1
            # Array brackets to correctly store each class vector as an array(vector) withing the larger array
            label_arr = np.array([class_vector])
        # Else append segments on the existing segment array
        else:
            average_val = np.average(data_segment.data)
            # If it is to little data for an entire segment(which is needed for it to be added to the array where all other arrays are of said length)
            # Fill it up with average
            while(data_segment.data.shape[0] < (window_size*1000 + 1)):
                data_segment.data = np.append(data_segment.data, average_val)

            segment_arr = np.append(segment_arr, [data_segment.data], axis = 0)

            # Figure out majoirty class of that segment
            # Create vector of zeros, majority class position is set to 1 since this segment belongs to said class
            # Add class vector to array of labels corresponding to each segment
            majority_class = round(np.average(label_segment.data))
            class_vector = np.zeros(num_classes)
            class_vector[majority_class] = 1
            label_arr = np.append(label_arr, [class_vector], axis = 0)


        # increment with window size-overlap
        # Break if next increment would contain samples that are outside the data
        t = t + (window_size-overlap)
        if ((t*1000) >= (data_length-window_size)):
            break


    return segment_arr, label_arr
#==============================================================================