import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import chirp, correlate
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d

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
recording = sd.rec(sample_rate*rec_duration, samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()

recording = recording.flatten()  # Flatten to 1D array if necessary
np.save(f"Data_files/{symbol_count}symbol_recording_to_test_with_w_noise.npy", recording)

#  Using saved recording
# recording = np.load(f"Data_files/{symbol_count}symbol_recording_to_test_with_w_noise.npy")

# STEP 2: initially synchronise

# The matched_filter_output has been changed to full as this is the default method and might be easier to work with
# this means that the max index detected is now at the end of the chirp
matched_filter_output = correlate(recording, chirp_sig, mode='full')

# Create plots of the recording and matched filter response note that the x axes are different. 
t_rec = np.arange(0, len(recording))
t_mat = np.arange(0, len(matched_filter_output))
# fig, (ax1, ax2) = plt.subplots(2, 1)

# ax1.plot(t_rec, recording, label='Recording', color='b')
# ax1.set_xlabel('X-axis 1')
# ax1.set_ylabel('Y-axis 1')
# ax1.legend()
# ax1.set_title('First Plot')

# ax2.plot(t_mat, matched_filter_output, label='Matched filter output', color='r')
# ax2.set_xlabel('X-axis 2')
# ax2.set_ylabel('Y-axis 2')
# ax2.legend()
# ax2.set_title('Second Plot')

# plt.tight_layout()

# plt.plot(abs(matched_filter_output))
# plt.show()

detected_index = np.argmax(matched_filter_output)
print(detected_index)

# Use matched filter to take out the chirp from the recording
chirp_fft = fft(chirp_sig)
n = int(sample_rate*chirp_duration)   # number of samples of the chirp 
detected_chirp = recording[detected_index-n:detected_index]
detected_fft = fft(detected_chirp)
channel_fft = detected_fft/chirp_fft
channel_impulse = ifft(channel_fft)

# channel impulse before resynchronisation
# plt.plot(abs(channel_impulse))  
# plt.show()

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
    print(impulse_start)
    if impulse_start > len(channel_impulse) / 2:
        impulse_start = impulse_start - len(channel_impulse)
    print(impulse_start)
    return impulse_start


def impulse_start_smoothing(channel_impulse):

    # Dataset
    x = np.arange(0, len(channel_impulse), 1)
    y = abs(channel_impulse)

    X_Y_Spline = make_interp_spline(x, y)

    # Returns evenly spaced numbers
    # over a specified interval.
    X_ = np.linspace(x.min(), x.max(), len(channel_impulse))
    Y_ = X_Y_Spline(X_)

    # Plotting the Graph
    plt.plot(X_, Y_)
    plt.title("Plot Smooth Curve Using the scipy.interpolate.make_interp_spline() Class")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def impulse_start_guassian(channel_impulse): 
    data = channel_impulse

    # Apply Gaussian smoothing
    # sigma determines the standard deviation of the Gaussian kernel
    # window size +- 3 data points can be interpreted as sigma = 1 for a standard Gaussian kernel
    smoothed_data = gaussian_filter1d(data, sigma=1)

    print("Original Data: ", data)
    print("Smoothed Data: ", smoothed_data)

impulse_shift = impulse_start_max(channel_impulse)
# impulse_shift = 0

#Recalculate the section of chirp we want
detected_chirp = recording[detected_index-n+impulse_shift:detected_index+impulse_shift]
detected_fft = fft(detected_chirp)
channel_fft = detected_fft/chirp_fft
channel_impulse = ifft(channel_fft)

# take the channel that is the length of the cyclic prefix, zero pad to get datachunk length and fft
channel_impulse_cut = channel_impulse[:prefix_len]
channel_impulse_full = list(channel_impulse_cut) + [0]*int(datachunk_len-prefix_len) # zero pad to datachunk length
channel_coefficients = fft(channel_impulse_full)

# plt.plot(abs(channel_impulse))
# plt.show()
# plt.plot(abs(channel_coefficients))
# plt.show()

# STEP 4: crop audio file to the data
data_start_index = detected_index+impulse_shift
recording_without_chirp = recording[data_start_index : data_start_index+recording_data_len]
# load in the file sent to test against
source_mod_seq = np.load(f"Data_files/mod_seq_{symbol_count}symbols.npy")[num_known_symbols*num_data_bins:]
print(len(source_mod_seq))


# STEP 5: cut into different blocks and get rid of cyclic prefix

num_symbols = int(len(recording_without_chirp)/symbol_len)  # Number of symbols 

print(f"Num of OFDM symbols: {num_symbols}")

time_domain_datachunks = np.array(np.array_split(recording_without_chirp, num_symbols))[:, prefix_len:]

sent_signal = np.load(f'Data_files/{symbol_count}symbol_overall_w_noise.npy')
sent_without_chirp = sent_signal[-symbol_count*symbol_len:]
sent_datachunks = np.array(np.array_split(sent_without_chirp, symbol_count))[:, prefix_len:]

ofdm_datachunks = fft(time_domain_datachunks)  # Does the fft of all symbols individually 
# channel_estimate = ofdm_datachunks[0]/fft(sent_datachunks[0])

def estimate_channel_from_known_ofdm(_num_known_symbols):
    channel_estimates = np.zeros((_num_known_symbols, datachunk_len), dtype='complex')
    for i in range(_num_known_symbols):
        channel_fft = ofdm_datachunks[i]/fft(sent_datachunks[i])
        channel_estimates[i] = channel_fft
    
    average_channel_estimate = np.mean(channel_estimates, axis=0)
    print(channel_estimates.shape)
    print(average_channel_estimate.shape)
    return average_channel_estimate

channel_estimate = estimate_channel_from_known_ofdm(num_known_symbols)

ofdm_datachunks = ofdm_datachunks[num_known_symbols:]/channel_estimate # Divide each value by its corrosponding channel fft coefficient. 
data = ofdm_datachunks[:, lower_bin:upper_bin+1] # Selects the values from 1 to 511

# data = data.flatten()
# data = data[:len(source_mod_seq)]  # as the binary data isn't an exact multiple of 511*2 we have zero padded this gets rid of zeros

# makes list of colours corresponding to the original modulated data



mult = 20

colors = np.where(source_mod_seq == mult*(1+1j), "b", #"b"
            np.where(source_mod_seq == mult*(-1+1j), "c", #"c"
            np.where(source_mod_seq == mult*(-1-1j), "m", #"m"
            np.where(source_mod_seq == mult*(1-1j), "y",  #"y"
            "Error"))))


# for mask_col in ["b", "c", "m", "y"]:
# mask_col = "b"
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
# Generate and plot data for each subplot
for i in range(2):
    for j in range(5):
        # Generate random data
        # index = 10*(i*5 + j) + 9
        index = i*5 + j
        _data = data[index]
        x = _data.real
        y = _data.imag
        _colors = colors[index*num_data_bins:(index+1)*num_data_bins]
        _source_mod_seq = source_mod_seq[index*num_data_bins:(index+1)*num_data_bins]
        # Plot on the corresponding subplot
        ax = axes[i, j]
        # ax.scatter(x[_colors==mask_col], y[_colors==mask_col], c = _colors[_colors==mask_col])
        ax.scatter(x, y, c = _colors)
        ax.axvline(0)
        ax.axhline(0)
        ax.set_xlim((-50,50))
        ax.set_ylim((-50,50))
        ax.set_aspect('equal')

        errors = 0
        for n, val in enumerate(_data):
            sent = _source_mod_seq[n]
            if val.real/sent.real < 0 or val.imag/sent.imag < 0:
                errors = errors + 1

        ax.set_title(f'OFDM Symbol {index + 1}\nerror % = {round(errors*100/len(_data), 2)}')
        # ax.text(10, 10, f"error % = {errors/len(_data)}")

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# plt.scatter(data.real, data.imag, c=colors)
# plt.axvline(0)
# plt.axhline(0)
# plt.show()



# step 6: map each value to bits using QPSK decision regions
# step 8: decode recieved bits to information bits
# step 9: convert information bits to file using standardised preamble.

# recovered_values = np.where(data.real >= 0 and data.imag >= 0, 1+1j, 
#             np.where(data.real < 0 and data.imag >= 0, -1+1j, 
#             np.where(data.real < 0 and data.imag < 0, -1-1j, 
#             np.where(data.real >= 0 and data.imag < 0, 1-1j, 
#             "Error"))))

# errors = np.count_nonzero(recovered_values-source_mod_seq)
# print(errors/len(recovered_values))

