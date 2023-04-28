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

    segment_arr = None
    label_arr = None


    m = pep.wrappers.EMGMeasurement(data, hz = sampling_rate)
    l = pep.wrappers.EMGMeasurement(labels, hz = sampling_rate)
    print(m)
    print(l)

    temp_data = m.apply_segmenter(beg_ts = 0, end_ts = 0.15)
    temp_label = l.apply_segmenter(beg_ts = 0, end_ts = 0.15)
    print(temp_data)
    print(temp_label)

    data_length = data.shape[0]
    if data_length < window_size:
        raise Exception("Data is smaller than window size")
    # Segment data with overlapping windows
    for t in range(0, data_length-window_size, window_size-overlap):
        print(data)
        print(labels)
        print(data.shape)
        print(labels.shape)
        print(window_size)
        print(t)
        temp_data = m.apply_segmenter(beg_ts = 0+t, end_ts = window_size+t)
        temp_label = l.apply_segmenter(beg_ts = 0+t, end_ts = window_size+t)
        print(data)
        print(labels)

        # Create array first time we run
        if t == 0:
            print(temp_data)
            segment_arr = np.array(temp_data)

            # Figure out majoirty class of that segment
            # Create vector of zeros, majority class position is set to 1 since this segment belongs to said class
            # Add class vector to array of labels corresponding to each segment
            print(temp_label)
            majority_class = round(np.average(temp_label))
            class_vector = np.zeros(num_classes)
            class_vector[majority_class] = 1
            label_arr = np.array(class_vector)
        # Else append segments on the existing segment array
        else:
            segment_arr = np.append(segment_arr, [temp_data], axis = 0)

            # Figure out majoirty class of that segment
            # Create vector of zeros, majority class position is set to 1 since this segment belongs to said class
            # Add class vector to array of labels corresponding to each segment
            majority_class = round(np.average(temp_label))
            class_vector = np.zeros(num_classes)
            class_vector[majority_class] = 1
            label_arr = np.append(label_arr, [class_vector], axis = 0)


    return segment_arr, label_arr
#==============================================================================
