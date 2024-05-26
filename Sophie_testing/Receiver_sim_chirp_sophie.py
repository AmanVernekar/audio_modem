import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import chirp, correlate
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d

import parameters

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
ch_sample_1 = parameters.ch_sample_1
ch_sample_2 = parameters.ch_sample_2
ch_len = parameters.ch_len

known_chirp_start = sample_rate + prefix_len


start = 1
r = 0.99
simulated_channel = start * r**np.arange(prefix_len) # + np.random.normal(0, noise_std, prefix_len)
plt.plot(simulated_channel)
plt.title("Simulated channel impulse")
plt.show()

channel_fft = fft(simulated_channel)

plt.plot(abs(channel_fft))
plt.title("channel fft")
plt.show()


# STEP 1: Generate transmitted chirp and record signal

t_total = np.linspace(0, rec_duration, int(sample_rate * rec_duration), endpoint=False)
t_chirp = np.linspace(0, chirp_duration, int(sample_rate * chirp_duration), endpoint=False)

chirp_sig = chirp(t_chirp, f0=chirp_start_freq, f1=chirp_end_freq, t1=chirp_duration, method=chirp_type)
chirp_sig = list(chirp_sig)

sent_signal = np.load(f'Sophie_testing/{symbol_count}symbol_overall.npy')
noise_std = 0.05
recording = np.convolve(sent_signal, simulated_channel, 'full')[:-prefix_len+1]
# recording = recording_without_noise + np.random.normal(0, noise_std, len(recording_without_noise))

plt.plot(recording)
plt.title("Simulated recording")
plt.show()

# STEP 2: initially synchronise
# Use matched filter to take out the chirp from the recording
chirp_fft = fft(chirp_sig)

plt.plot(abs(chirp_fft))
plt.title("fft of chirp")
plt.show()

n = int(sample_rate*chirp_duration)   # number of samples of the chirp 
detected_chirp = recording[known_chirp_start:known_chirp_start+n]

plt.plot(detected_chirp)
plt.title("Chirp from recording")
plt.show()

# detected_fft = fft(detected_chirp)

# plt.plot(detected_fft)
# plt.title("FFT of detected chirp")
# plt.show()

detected_fft = fft(detected_chirp)
channel_coefficients = np.zeros((sample_rate*chirp_duration), dtype = complex)
channel_coefficients[ch_sample_1:ch_sample_2+1] = detected_fft[ch_sample_1:ch_sample_2+1]/chirp_fft[ch_sample_1:ch_sample_2+1]
channel_coefficients[ch_len-ch_sample_2:ch_len-ch_sample_1+1] = detected_fft[ch_len-ch_sample_2:ch_len-ch_sample_1+1]/chirp_fft[ch_len-ch_sample_2:ch_len-ch_sample_1+1]

plt.plot(abs(detected_fft))
plt.title("FFT of recorded chirp")
plt.show() 

# channel_coefficients = detected_fft/chirp_fft

plt.plot(abs(channel_coefficients))
plt.title("channel coefficents")
plt.show()

# th_element = ch_sample_2-ch_sample_1/datachunk_len
# print(th_element)
# channel_coefficients_section = channel_coefficients[::th_element]

channel_impulse = ifft(channel_coefficients)

plt.plot(abs(channel_impulse))
plt.title("channel_impulse")
plt.show()

channel_impulse = ifft(channel_coefficients)[:prefix_len]
channel_impulse_full = list(channel_impulse) + [0]*int((datachunk_len-prefix_len))
channel_coefficients = fft(channel_impulse_full)

plt.plot(channel_impulse)
plt.title("channel_impulse cut")
plt.show()



# Use known OFDM and known synchronisation:
# fft_first_ofdm_symbol_sent = fft(sent_signal[sample_rate + prefix_len:sample_rate + prefix_len + datachunk_len])

# plt.plot(fft_first_ofdm_symbol_sent)
# plt.title("fft of sent ofdm symbol")
# plt.show()

# first_ofdm_symbol_rec = recording[known_data_start:known_data_start+datachunk_len]
# first_ofdm_symbol_rec_fft = fft(first_ofdm_symbol_rec)

# plt.plot(first_ofdm_symbol_rec_fft)
# plt.title("fft of recorded ofdm symbol")
# plt.show()

# channel_coefficients = first_ofdm_symbol_rec_fft/ fft_first_ofdm_symbol_sent
# channel_impulse = ifft(channel_coefficients)

# plt.plot(channel_coefficients)
# plt.title("channel coefficients")
# plt.show()

# channel impulse before resynchronisation
# plt.plot((channel_impulse))
# plt.title("Channel impulse response")
# plt.show()
# plt.title("Channel coefficients")
# plt.plot(abs(channel_fft))
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
recording_without_chirp = recording[(sample_rate*6)+(2*prefix_len): ]
plt.plot(recording_without_chirp)
plt.title("Rec of data")
plt.show()
# load in the file sent to test against
source_mod_seq = np.load(f"Sophie_testing/mod_seq_{symbol_count}symbols.npy")
print(len(source_mod_seq))


# STEP 5: cut into different blocks and get rid of cyclic prefix

num_symbols = int(len(recording_without_chirp)/symbol_len)  # Number of symbols 

print(f"Num of OFDM symbols: {num_symbols}")

time_domain_datachunks = np.array(np.array_split(recording_without_chirp, num_symbols))[:, prefix_len:]

ofdm_datachunks = fft(time_domain_datachunks)  # Does the fft of all symbols individually 
ofdm_datachunks = ofdm_datachunks/channel_coefficients # Divide each value by its corrosponding channel fft coefficient. 
data = ofdm_datachunks[:, lower_bin:upper_bin+1] # Selects the values from 1 to 511

data = data.flatten()
data = data[:len(source_mod_seq)]  # as the binary data isn't an exact multiple of 511*2 we have zero padded this gets rid of zeros

# makes list of colours corresponding to the original modulated data
colors = np.where(source_mod_seq == 1+1j, "b", 
            np.where(source_mod_seq == -1+1j, "c", 
            np.where(source_mod_seq == -1-1j, "m", 
            np.where(source_mod_seq == 1-1j, "y", 
            "Error"))))

# plots the received data with colour corresponding to the sent data. 
plt.scatter(data.real, data.imag, c=colors)
plt.show()



# step 6: map each value to bits using QPSK decision regions
# step 8: decode recieved bits to information bits
# step 9: convert information bits to file using standardised preamble.
