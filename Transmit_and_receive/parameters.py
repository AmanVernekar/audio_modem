import numpy as np

#-----------------------------------------------------
# Global parameters
#-----------------------------------------------------

prefix_len = 1024                            # length of cyclic prefix
sample_rate = 48000                          # samples per second

rec_duration = 19                            # duration of recording in seconds

#-----------------------------------------------------
# Chirp parameters
#-----------------------------------------------------

chirp_start_freq = 761.72                    # chirp start freq (Hz)
chirp_end_freq = 8824.22                     # chirp end freq (Hz)
chirp_type = "linear"                        # chirp type
chirp_duration = 1.365                       # duration of chirp in seconds

#-----------------------------------------------------
# OFDM parameters
#-----------------------------------------------------

datachunk_len = 4096                        # length of the data in the OFDM symbol (DFT length)
lower_bin = 85
upper_bin = 733

symbol_len = datachunk_len + prefix_len     # total length of symbol

#-----------------------------------------------------
# LDPC parameters
#-----------------------------------------------------

ldpc_type = '802.16'
ldpc_rate = '1/2'
ldpc_z = 54
ldpc_k = 648

#-----------------------------------------------------
# Parameters for testing
#-----------------------------------------------------

symbol_count = 105
recording_data_len = symbol_count * symbol_len