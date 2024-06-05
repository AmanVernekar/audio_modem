import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import chirp, correlate
from math import sqrt
from ldpc_jossy.py import ldpc
from transmit_file_data import encoded_binary_to_ofdm_datachunk


import parameters
import our_chirp

datachunk_len = parameters.datachunk_len             # length of the data in the OFDM symbol
prefix_len = parameters.prefix_len                   # length of cyclic prefix
symbol_len = parameters.symbol_len                   # total length of symbol
sample_rate = parameters.sample_rate                 # samples per second
rec_duration = parameters.rec_duration               # duration of recording in seconds
recording_data_len = parameters.recording_data_len   # number of samples expected in recording
chirp_duration = parameters.chirp_duration           # duration of chirp in seconds
chirp_start_freq = parameters.chirp_start_freq       # chirp start freq
chirp_end_freq = parameters.chirp_end_freq           # chirp end freq
chirp_type = parameters.chirp_type                   # chirp type
lower_bin = parameters.lower_bin
upper_bin = parameters.upper_bin
num_data_bins = upper_bin-lower_bin+1
num_known_symbols = 1
chirp_samples = int(sample_rate * chirp_duration)
known_datachunk = parameters.known_datachunk
known_datachunk = known_datachunk.reshape(1, 4096)
alpha = 0.1

# STEP 1: Generate transmitted chirp and record signal
chirp_sig = our_chirp.chirp_sig

# Determines if we record in real life or get file which is already recorded
do_real_recording = False
 
if do_real_recording:
    # Using real recording
    recording = sd.rec(sample_rate*rec_duration, samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()

    recording = recording.flatten()  # Flatten to 1D array if necessary
    np.save(f"Data_files/example_file_recording_to_test_with.npy", recording)

else:
    # Using saved recording
    recording = np.load(f"Data_files/example_file_recording_to_test_with.npy")

# STEP 2: Initial Synchronisation

# The matched_filter_output has been changed to full as this is the default method and might be easier to work with
# this means that the max index detected is now at the end of the chirp

matched_filter_output = correlate(recording, chirp_sig, mode='full')
matched_filter_first_half = matched_filter_output[:int(len(matched_filter_output)/2)]

detected_index = np.argmax(matched_filter_first_half)
print(f"The index of the matched filter output is {detected_index}")


# Re-sync off for now
do_cyclic_resynch = False

if do_cyclic_resynch:
    # Use matched filter to take out the chirp from the recording
    def calculated_impulse_from_chirp(detected_index):
        chirp_fft = fft(chirp_sig)
        detected_chirp = recording[detected_index-chirp_samples:detected_index]
        detected_fft = fft(detected_chirp)
        channel_fft = detected_fft/chirp_fft
        channel_impulse = ifft(channel_fft)

        return channel_impulse

    # STEP 3: resynchronise and compute channel coefficients from fft of channel impulse response 
    # functions to choose the start of the impulse
    def impulse_start_10_90_jump(channel_impulse):   
        channel_impulse_max = np.max(channel_impulse)
        channel_impulse_10_percent = 0.1 * channel_impulse_max
        channel_impulse_90_percent = 0.6 * channel_impulse_max

        impulse_start = 0

        for i in range(len(channel_impulse) - 1):
            if channel_impulse[i] < channel_impulse_10_percent and channel_impulse[i + 5] > channel_impulse_90_percent:
                impulse_start = i + 5
                break

        if impulse_start > len(channel_impulse) / 2:
            impulse_start = impulse_start - len(channel_impulse)

        return impulse_start

    def impulse_start_max(channel_impulse):
        impulse_start = np.argmax(abs(channel_impulse))
        # print(impulse_start)
        if impulse_start > len(channel_impulse) / 2:
            impulse_start = impulse_start - len(channel_impulse)
        # print(impulse_start)
        return impulse_start

    channel_impulse = calculated_impulse_from_chirp(detected_index)
    impulse_shift = impulse_start_max(channel_impulse)
else:
    impulse_shift = 0




# Process through which we calculate optimal shift with error rates from known OFDM symbols
# Off for now
optimisation_resynch = False

if optimisation_resynch:
    shifts = range(-100,100)
    total_errors = np.zeros((len(shifts)))  

    for g, shift in enumerate(shifts):

        # Are we indexing correctly?
        data_start_index = detected_index+shift+prefix_len
        recording_without_chirp = recording[data_start_index : data_start_index+recording_data_len]

        # STEP 5: cut into different blocks and get rid of cyclic prefix
        num_symbols = int(len(recording_without_chirp)/symbol_len)  # Number of symbols 
        if g == 0: 
            print("num_symbols_calc: ", num_symbols)
        time_domain_datachunks = np.array(np.array_split(recording_without_chirp, num_symbols))[:, prefix_len:]
        ofdm_datachunks = fft(time_domain_datachunks)  # Does the fft of all symbols individually 

        channel_estimate = ofdm_datachunks[0]/known_datachunk[0]  
        # Why don't we get divide by zero error here? middle known datachunk val = 0 !!!

        ofdm_datachunks = ofdm_datachunks[num_known_symbols:]/channel_estimate # Divide each value by its corrosponding channel fft coefficient. 
        data = ofdm_datachunks[:, lower_bin:upper_bin+1] # Selects the values from 1 to 511

        total_error = 0

        _data = data[:num_known_symbols].flatten()
        _source_mod_seq = source_mod_seq[:num_known_symbols * num_data_bins]
        for w, value in enumerate(_data):
            sent = _source_mod_seq[w]
            if value.real/sent.real < 0 or value.imag/sent.imag < 0:
                        total_error = total_error + 1

        total_errors[g] = total_error*10/len(_data)

    plt.plot(shifts, total_errors)
    plt.axvline(shifts[np.argmin(total_errors)])
    plt.ylabel("Bit error percentage (%)")
    plt.xlabel("Index")
    # plt.show()


    best_shift = shifts[np.argmin(total_errors)]
    print(best_shift)
    print(np.min(total_errors))
else:
    best_shift = 0

# STEP:
# Find the data from the best shift. 
data_start_index = detected_index+best_shift+prefix_len  # Again are we sure this is indexed correctly?
recording_without_chirp = recording[data_start_index : data_start_index+recording_data_len]  # Again are we sure this is indexed correctly?

num_symbols = int(len(recording_without_chirp)/symbol_len)  # Number of symbols detected - is this correct? what if phase drift?
time_domain_datachunks = np.array(np.array_split(recording_without_chirp, num_symbols))[:, prefix_len:]
ofdm_datachunks = fft(time_domain_datachunks)  # Does the fft of all symbols individually 

channel_estimate = ofdm_datachunks[0]/known_datachunk[0]
# Why don't we get divide by zero error here? middle known datachunk val = 0 !!!

_ofdm_datachunks = ofdm_datachunks[num_known_symbols:]/channel_estimate # Divide each value by its corrosponding channel fft coefficient. 
complex_data = _ofdm_datachunks[:, lower_bin:upper_bin+1] # Selects the values from 1 to 511

complex_data.flatten() # convert data to 1 dimensional array of complex values

# STEP
# detect binary from complex values

def complex_vals_to_binary_hard_decision(complex_values):
    """Applies hard decision boundaries to get binary array"""
    binary_array = [] # placeholder
    return binary_array

def do_ldpc(complex_values):
    """Uses LDPC to go from complex values from 1 received OFDM symbol to the information data"""
    hard_binary_data, soft_binary_data = (0,0) #placeholder
    return hard_binary_data, soft_binary_data

# STEP
# calculate the performance of the receiver

# -------------------------------------------------------------------------------------------------------------
# Please look at this bit I don't get it

# Are we indexing this correctly? Accounts for chirp etc?
# I'm guessing this is the sent signal when modulated to complex values?
source_mod_seq = np.load(f"Data_files/mod_seq_example_file.npy")[num_known_symbols*num_data_bins:]

# I'm guessing this is the sent signal as binary array? 
sent_signal = np.load(f'Data_files/example_file_overall_sent.npy')


# A lot of this might be in parameters. 
data_start_sent_signal = sample_rate + (prefix_len*2) + (chirp_samples)
end_start_sent_signal = (prefix_len*2) + (chirp_samples)
sent_without_chirp = sent_signal[data_start_sent_signal: - end_start_sent_signal ]
print("sent data length", len(sent_without_chirp))
num_symbols = int(len(sent_without_chirp)/symbol_len)
recording_data_len = num_symbols * symbol_len
print("num of symbols: ", num_symbols)
sent_datachunks = np.array(np.array_split(sent_without_chirp, num_symbols))[:, prefix_len:]

colors = np.where(source_mod_seq == (1+1j), "b", 
            np.where(source_mod_seq == (-1+1j), "c", 
            np.where(source_mod_seq == (-1-1j), "m", 
            np.where(source_mod_seq == (1-1j), "y", 
            "Error"))))

# -------------------------------------------------------------------------------------------------------------