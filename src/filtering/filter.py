#!/usr/bin/env python3
#==============================================================================
# Author: Carl Larsson
# Description: Performs filtering
# Date: 2023-04-25
#==============================================================================


import pyemgpipeline as pep
from scipy import signal


#==============================================================================
# Remove dc offset
def rm_offset(data, sampling_rate):
    m = pep.wrappers.EMGMeasurement(data, hz=sampling_rate)
    m.apply_dc_offset_remover()

    return m.data
#==============================================================================
# Performs bandpass filtering
def bandpass(data, sampling_rate, filter_order = 4, low_cutoff = 20, high_cutoff = 450):
    m = pep.wrappers.EMGMeasurement(data, hz=sampling_rate)
    m.apply_bandpass_filter(bf_order=filter_order, bf_cutoff_fq_lo=low_cutoff, bf_cutoff_fq_hi=high_cutoff)

    return m.data
#==============================================================================
# Performs notch filtering
def notch(data, sampling_rate, notch_freq = 50, quality_factor = 30):
    # Design IIR notch filter
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, sampling_rate)

    # Return filtered signal
    return signal.filtfilt(b_notch, a_notch, data)
#==============================================================================
