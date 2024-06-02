import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import chirp, correlate

import parameters
import our_chirp

datachunk_len = parameters.datachunk_len             # length of the data in the OFDM symbol
prefix_len = parameters.prefix_len                   # length of cyclic prefix
symbol_len = parameters.symbol_len                   # total length of symbol
sample_rate = parameters.sample_rate                 # samples per second
rec_duration = parameters.rec_duration               # duration of recording in seconds
chirp_duration = parameters.chirp_duration           # duration of chirp in seconds
chirp_start_freq = parameters.chirp_start_freq       # chirp start freq
chirp_end_freq = parameters.chirp_end_freq           # chirp end freq
chirp_type = parameters.chirp_type                   # chirp type
recording_data_len = parameters.recording_data_len   # number of samples of data (HOW IS THIS FOUND)
lower_bin = parameters.lower_bin
upper_bin = parameters.upper_bin
symbol_count = parameters.symbol_count
num_data_bins = upper_bin-lower_bin+1
num_known_symbols = 5

# STEP 1: Generate transmitted chirp and record signal
chirp_sig = our_chirp.chirp_sig

# Using real recording 
# recording = sd.rec(sample_rate*rec_duration, samplerate=sample_rate, channels=1, dtype='float32')
# sd.wait()

# recording = recording.flatten()  # Flatten to 1D array if necessary
# np.save(f"Data_files/{symbol_count}symbol_recording_to_test_with_w_noise.npy", recording)

#  Using saved recording
recording = np.load(f"Data_files/{symbol_count}symbol_recording_to_test_with_w_noise.npy")

# STEP 2: initially synchronise

# The matched_filter_output has been changed to full as this is the default method and might be easier to work with
# this means that the max index detected is now at the end of the chirp
matched_filter_output = correlate(recording, chirp_sig, mode='full')

detected_index = np.argmax(matched_filter_output)
print(detected_index)

# Use matched filter to take out the chirp from the recording
chirp_fft = fft(chirp_sig)
chirp_sample_count = int(sample_rate*chirp_duration)   # number of samples of the chirp 
detected_chirp = recording[detected_index-chirp_sample_count:detected_index]
detected_fft = fft(detected_chirp)
channel_fft = detected_fft/chirp_fft
channel_impulse = ifft(channel_fft)

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


impulse_shift = impulse_start_max(channel_impulse)
impulse_shift = 0

shifts = range(-100,100)
total_errors = np.zeros((len(shifts)))

source_mod_seq = np.load(f"Data_files/mod_seq_{symbol_count}symbols.npy")[num_known_symbols*num_data_bins:]

sent_signal = np.load(f'Data_files/{symbol_count}symbol_overall_w_noise.npy')
sent_without_chirp = sent_signal[-symbol_count*symbol_len:]
sent_datachunks = np.array(np.array_split(sent_without_chirp, symbol_count))[:, prefix_len:]

colors = np.where(source_mod_seq == (1+1j), "b", 
            np.where(source_mod_seq == (-1+1j), "c", 
            np.where(source_mod_seq == (-1-1j), "m", 
            np.where(source_mod_seq == (1-1j), "y", 
            "Error"))))


def estimate_channel_from_known_ofdm(_num_known_symbols):
        channel_estimates = np.zeros((_num_known_symbols, datachunk_len), dtype='complex')
        for i in range(_num_known_symbols):
            channel_fft = ofdm_datachunks[i]/fft(sent_datachunks[i])
            channel_estimates[i] = channel_fft
        
        average_channel_estimate = np.mean(channel_estimates, axis=0)
        return average_channel_estimate

for g, shift in enumerate(shifts):

    data_start_index = detected_index+shift+prefix_len
    recording_without_chirp = recording[data_start_index : data_start_index+recording_data_len]

    # STEP 5: cut into different blocks and get rid of cyclic prefix
    num_symbols = int(len(recording_without_chirp)/symbol_len)  # Number of symbols 
    time_domain_datachunks = np.array(np.array_split(recording_without_chirp, num_symbols))[:, prefix_len:]
    ofdm_datachunks = fft(time_domain_datachunks)  # Does the fft of all symbols individually 

    channel_estimate = estimate_channel_from_known_ofdm(num_known_symbols)

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
plt.show()


best_shift = shifts[np.argmin(total_errors)]
print(best_shift)
print(np.min(total_errors))


# Refinding the data from the best shift. 
data_start_index = detected_index+best_shift+prefix_len
recording_without_chirp = recording[data_start_index : data_start_index+recording_data_len]

num_symbols = int(len(recording_without_chirp)/symbol_len)  # Number of symbols 
time_domain_datachunks = np.array(np.array_split(recording_without_chirp, num_symbols))[:, prefix_len:]
ofdm_datachunks = fft(time_domain_datachunks)  # Does the fft of all symbols individually 

channel_estimate = estimate_channel_from_known_ofdm(num_known_symbols)

ofdm_datachunks = ofdm_datachunks[num_known_symbols:]/channel_estimate # Divide each value by its corrosponding channel fft coefficient. 
data = ofdm_datachunks[:, lower_bin:upper_bin+1] # Selects the values from 1 to 511

first_data = data[0]
first_colours = colors[:num_data_bins]

plt.scatter(first_data.real, first_data.imag, c=first_colours)
plt.xlim(-20, 20)  # Limit the x-axis 
plt.ylim(-20, 20)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.show()


