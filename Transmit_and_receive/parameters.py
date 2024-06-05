import numpy as np

datachunk_len = 4096                        # length of the data in the OFDM symbol
prefix_len = 1024                            # length of cyclic prefix
symbol_len = datachunk_len + prefix_len     # total length of symbol
sample_rate = 48000                         # samples per second
rec_duration = 12                            # duration of recording in seconds
chirp_duration = 1.365                          # duration of chirp in seconds
chirp_start_freq = 761.72                     # chirp start freq
chirp_end_freq = 8824.22                      # chirp end freq
chirp_type = "linear"                       # chirp type
recording_data_len = 10240                  # number of samples of data (HOW IS THIS FOUND)
lower_bin = 85
upper_bin = 732
symbol_count = 105
chirp_reduction = 0.1
ldpc_z = 54
ldpc_k = 648



np.random.seed(1)
random_integers = np.random.randint(0,4,2047)
random_complex_values = np.sqrt(2) * np.exp(0 + random_integers * 1j * np.pi/2  + np.pi/4 * 1j)
known_datachunk = np.zeros((datachunk_len), dtype=complex)
known_datachunk[1 : datachunk_len//2] = random_complex_values
known_datachunk[datachunk_len//2  + 1:] = np.conjugate(random_complex_values[::-1])



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