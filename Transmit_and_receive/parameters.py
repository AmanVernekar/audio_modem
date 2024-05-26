import numpy as np

datachunk_len = 4096                        # length of the data in the OFDM symbol
prefix_len = 512                           # length of cyclic prefix
symbol_len = datachunk_len + prefix_len     # total length of symbol
sample_rate = 44100                         # samples per second
rec_duration = 19                            # duration of recording in seconds
chirp_duration = 5                          # duration of chirp in seconds
chirp_start_freq = 0.01                     # chirp start freq
chirp_end_freq = 22050                      # chirp end freq
chirp_type = "linear"                       # chirp type
recording_data_len = 483840                  # number of samples of data (HOW IS THIS FOUND)
lower_bin = 85
upper_bin = 850
symbol_count = 105

def calculate_bins(sample_rate, lower_freq, upper_freq, ofdm_chunk_length):
    lower_bin = np.ceil((lower_freq / sample_rate) * ofdm_chunk_length).astype(int)  # round up
    upper_bin = np.floor((upper_freq / sample_rate) * ofdm_chunk_length).astype(int)  # round down

    # print(f"""
    # for the parameters: sample rate = {sample_rate}Hz
    #                     information bandlimited to {lower_freq} - {upper_freq}Hz
    #                     OFDM symbol length = {ofdm_chunk_length}
    #             lower bin is {lower_bin}
    #             upper bin is {upper_bin}
    # """)
    return lower_bin, upper_bin