import numpy as np
import parameters

lower_bin = parameters.lower_bin
upper_bin = parameters.upper_bin
num_data_bins = upper_bin - lower_bin + 1
known_datachunk = parameters.known_datachunk

def reverse_phase(known_datachunk, received_datachunk):
    bits = np.zeros((num_data_bins,2))
    symbol_data = received_datachunk[lower_bin:upper_bin+1]
    known_datachunk_data_bins = known_datachunk[lower_bin:upper_bin+1]/(1+1j)
    # phase = np.angle(symbol_data/known_datachunk_data_bins) #+ np.pi # 0 to 2pi
    # phase = phase/(np.pi/4) # 0 to 8

    # for i, ph in enumerate(phase):
    #     if ph >= 7 and ph < 1:
    #         bits[i] = [0,0]
    #     elif ph >= 1 and ph < 3:
    #         bits[i] = [0,1]
    #     elif ph >= 3 and ph < 5:
    #         bits[i] = [1,1]
    #     elif ph >= 5 and ph < 7:
    #         bits[i] = [1,0]

    # bits = bits.flatten()
    # return bits
    return symbol_data/known_datachunk_data_bins

print(reverse_phase(known_datachunk, known_datachunk*np.exp(1j * 1.5*np.pi/2)))# == [0]*(num_data_bins*2))
np.any