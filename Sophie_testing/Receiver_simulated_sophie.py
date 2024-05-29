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

chirp_start = sample_rate + prefix_len
known_data_start = sample_rate + prefix_len
print(known_data_start)


start = 1
r = 0.99
simulated_channel = start * r**np.arange(prefix_len) # + np.random.normal(0, noise_std, prefix_len)
# plt.plot(simulated_channel)
# plt.title("Simulated channel impulse")
# plt.show()

chirp_sig = our_chirp.chirp_sig

sent_signal = np.load(f'Data_files/{symbol_count}symbol_overall_sent.npy')
noise_std = 0.05
recording_short = np.convolve(sent_signal, simulated_channel, 'full')[:-prefix_len+1]
#recording_short = recording_without_noise + np.random.normal(0, noise_std, len(recording_without_noise))
recording = list(recording_short) + [0]*sample_rate 

# plt.plot(recording)
# plt.title("Simulated recording")
# plt.show()

# STEP 2: initially synchronise

# Use matched filter to take out the chirp from the recording
chirp_fft = fft(chirp_sig)

# plt.plot(abs(chirp_fft))
# plt.title("Chirp fft")
# plt.show()

n = int(sample_rate*chirp_duration)   # number of samples of the chirp 
chirp_end = chirp_start + n 
detected_chirp = recording[chirp_start - 100 :chirp_end - 100]  # CHECK THIS. 
print(f"Cut out chirp length {len(detected_chirp)}")

plt.plot(detected_chirp)
plt.title("Chirp from recording")
plt.show()

detected_fft = fft(detected_chirp)

# plt.plot(abs(detected_fft))
# plt.title("Detected chirp fft")
# plt.show()

channel_estimate = detected_fft/chirp_fft

# plt.plot(abs(channel_fft))
# plt.title("Channel fft")
# plt.show()

channel_impulse = ifft(channel_estimate)

plt.plot(abs(channel_impulse))
plt.title("Channel impulse")
plt.show()

channel_impulse_cut = channel_impulse[:prefix_len]
channel_impulse_full = list(channel_impulse_cut) + [0]*int(datachunk_len-prefix_len) # zero pad to datachunk length
channel_estimate = fft(channel_impulse_full)

# plt.plot(abs(channel_estimate))
# plt.title("Channel coefficients")
# plt.show()

# Use known OFDM and known synchronisation:

data_start = chirp_end + prefix_len 
recording_without_chirp = recording[data_start + 10 : data_start + recording_data_len +10]
print(f"chirp end: {chirp_end}")
print(f"recording without chrip: {len(recording_without_chirp)}")
time_domain_datachunks = np.array(np.array_split(recording_without_chirp, symbol_count))[:, prefix_len:]
ofdm_datachunks = fft(time_domain_datachunks)

sent_signal = np.load(f'Data_files/{symbol_count}symbol_overall_w_noise.npy')
sent_without_chirp = sent_signal[data_start : data_start + recording_data_len]
sent_datachunks = np.array(np.array_split(sent_without_chirp, symbol_count))[:, prefix_len:]

def estimate_channel_from_known_ofdm(_num_known_symbols):
    channel_estimates = np.zeros((_num_known_symbols, datachunk_len), dtype='complex')
    for i in range(_num_known_symbols):
        channel_fft = ofdm_datachunks[i]/fft(sent_datachunks[i])
        channel_estimates[i] = channel_fft
    
    average_channel_estimate = np.mean(channel_estimates, axis=0)
    print(channel_estimates.shape)
    print(average_channel_estimate.shape)
    return average_channel_estimate

# channel_estimate = estimate_channel_from_known_ofdm(num_known_symbols)


# plt.plot(abs(channel_estimate))
# plt.title("channel coefficients")
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

# impulse_shift = impulse_start_max(channel_impulse)

# Recalculate the section of chirp we want
# detected_chirp = recording[detected_index-n+impulse_shift:detected_index+impulse_shift]
# detected_fft = fft(detected_chirp)
# channel_fft = detected_fft/chirp_fft
# channel_impulse = ifft(channel_fft)

# take the channel that is the length of the cyclic prefix, zero pad to get datachunk length and fft
# channel_impulse_cut = channel_impulse[:prefix_len]
# channel_impulse_full = list(channel_impulse_cut) + [0]*int(datachunk_len-prefix_len) # zero pad to datachunk length
# channel_coefficients = fft(channel_impulse_full)

# plt.plot(abs(channel_impulse_full))
# plt.title("Channel impulse")
# plt.show()

# STEP 4: crop audio file to the data
#data_start_index = detected_index  +impulse_shift
# load in the file sent to test against
source_mod_seq = np.load(f"Data_files/mod_seq_{symbol_count}symbols.npy")[num_known_symbols*num_data_bins:]
print(len(source_mod_seq))

ofdm_datachunks = ofdm_datachunks[num_known_symbols:]/channel_estimate # Divide each value by its corrosponding channel fft coefficient. 
data = ofdm_datachunks[:, lower_bin:upper_bin+1]


colors = np.where(source_mod_seq == 1+1j, "b", #"b"
            np.where(source_mod_seq == -1+1j, "c", #"c"
            np.where(source_mod_seq == -1-1j, "m", #"m"
            np.where(source_mod_seq == 1-1j, "y",  #"y"
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



# step 6: map each value to bits using QPSK decision regions
# step 8: decode recieved bits to information bits
# step 9: convert information bits to file using standardised preamble.
