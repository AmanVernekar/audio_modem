# lets use frequency bins for information of 1kHz to 8kHz (approximately)
# 44.1 kHz sampling rate
# => maybe create a function to evaluate correct bins? from lower cutoff, upper cutoff, sampling freq

import numpy as np

#in this transmitter program:
#step 0: encode file as binary data (to do later on)
#step 1: encode binary data as complex symbols using QPSK
#step 2: insert QPSK symbols into as many OFDM symbols as required (only in correct numbers of bins)
#  - this should ensure that the OFDM symbol has conjugate symmetry
#step 3: DFT each OFDM symbol
#step 4: add cyclic prefix to each part
#step 5: concatenate all time domain blocks. Do we want to space out these blocks? beyond just the prefix? (probs no)
#step 6:convert to audio file to transmit across channel. add chirp beforehand etc.

#for separate decoder program:
#step 1: perform channel estimation and synchronisation steps
#step 2: crop audio file to the data
#step 3: cut into different blocks and get rid of cyclic prefix
#step 4: take the DFT
#step 5: divide by channel coefficients determined in step 1
#step 6: choose complex values corresponding to information bits
#step 7: map each value to bits using QPSK decision regions
#step 8: decode recieved bits to information bits
#step 9: convert information bits to file using standardised preamble.
#for testing: evaluate performance using percentage of bits successfully detected.

information_sequence = np.array([1,0,1])
decoded_information = np.array([1,0,0])

accuracy = np.mean(information_sequence == decoded_information) *100  # percentage of matching bits
print(f"Percentage of accurately detected bits: {accuracy:.5f}%")