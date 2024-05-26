import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import chirp, correlate
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d


from . import parameters


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


start = 1
r = 0.99
noise_std = 0.05
simulated_channel = start * r**np.arange(prefix_len) + np.random.normal(0, noise_std, prefix_len)

sent_signal = np.load(f'{symbol_count}symbol_overall.npy')
recording = np.convolve(sent_signal, simulated_channel, 'full')[:-(prefix_len-1)]
recording = recording + np.random.normal(0, 0.025, len(recording))
recording_without_chirp = recording[-(symbol_count*symbol_len):]
sent_without_chirp = sent_signal[-symbol_count*symbol_len:]

time_domain_datachunks = np.array(np.array_split(recording_without_chirp, symbol_count))[:, prefix_len:]
# sent_datachunks = np.array(np.array_split(sent_without_chirp, symbol_count))[:, prefix_len:]
# channel_fft = fft(time_domain_datachunks[0])/fft(sent_datachunks[0])


t_chirp = np.linspace(0, chirp_duration, int(sample_rate * chirp_duration), endpoint=False)
chirp_sig = chirp(t_chirp, f0=chirp_start_freq, f1=chirp_end_freq, t1=chirp_duration, method=chirp_type)
received_chirp = recording[sample_rate+prefix_len:(chirp_duration+1)*sample_rate+prefix_len]
plt.plot(chirp_sig, label='sent chirp')
plt.plot(received_chirp, label='rec chirp')
plt.legend()
plt.show()
plt.plot(received_chirp - chirp_sig)
plt.show()
channel_fft = fft(received_chirp)/fft(chirp_sig)
channel = ifft(channel_fft)[:datachunk_len]
plt.plot(np.abs(channel))
plt.show()
channel_fft = fft(channel)


ofdm_datachunks = fft(time_domain_datachunks)  # Does the fft of all symbols individually 
ofdm_datachunks = ofdm_datachunks/channel_fft # Divide each value by its corrosponding channel fft coefficient. 
data = ofdm_datachunks[:, lower_bin:upper_bin+1] # Selects the values from 1 to 511
data = data.flatten()


source_mod_seq = np.load(f"mod_seq_{symbol_count}symbols.npy")
colors = np.where(source_mod_seq == 1+1j, "b", 
            np.where(source_mod_seq == -1+1j, "c", 
            np.where(source_mod_seq == -1-1j, "m", 
            np.where(source_mod_seq == 1-1j, "y", 
            "Error"))))

# plots the received data with colour corresponding to the sent data. 
plt.scatter(data.real, data.imag, c=colors)
plt.show()
