#!/usr/bin/env python3

import serial



#==============================================================================
# Function and class for reading data
# Class for reading data
ser = serial.Serial("/dev/ttyACM1")
class ReadLine:
    def __init__(self, s):
        self.buf = bytearray()
        self.s = s
    def readline(self):
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[:i+1]
            self.buf = self.buf[i+1:]
            return r
        while True:
            i = max(1, min(2048, self.s.sin_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buf + data[:i+1]
                self.buf[0:] = data[i+1:]
                return r
            else:
                self.buf.extend(data)
#==============================================================================
# Function for reading data
def load_data(number_samples_to_load):
    read_voltage_from_adc = ReadLine(ser)
    sample_counter = 0
    buf = []
    while sample_counter < number_samples_to_load:
        read_voltage_from_adc = ser.readline()
        read_voltage_from_adc = read_voltage_from_adc.decode('utf-8').rstrip('\n').rstrip('\r')
        if read_voltage_from_adc != "" and '\r' not in str(read_voltage_from_adc):
            #print(read_voltage_from_adc)
            buf.append(int(read_voltage_from_adc))
            sample_counter += 1
    return buf
