#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Performs filtering
# Date: 2023-04-25
#==============================================================================


import pyemgpipeline as pep


#==============================================================================
# Remove dc offset
def rm_offset(data, sampling_rate):
    hz = sampling_rate
    m = pep.wrappers.EMGMeasurement(data, hz=hz)
    m.apply_dc_offset_remover()

    return m.data
#==============================================================================
# Performs bandpass filtering
def bandpass(data, sampling_rate):
    hz = sampling_rate
    m = pep.wrappers.EMGMeasurement(data, hz=hz)
    m.apply_bandpass_filter(hz = sampling_rate, bf_order=4, bf_cutoff_fq_lo=20, bf_cutoff_fq_hi=500)

    return m.data
#==============================================================================
# Performs notch filtering
def notch(data, sampling_rate):
    pass
#==============================================================================
