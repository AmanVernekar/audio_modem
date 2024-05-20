import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt


datachunk_len = 1024
prefix_len = 32
symbol_len = datachunk_len + prefix_len
lower_bin = 24
upper_bin = 185


recording_without_chirp = np.array([])
channel_fft = np.array([])
source_data = np.array([])

# step 1: 
# step 2: perform channel estimation and synchronisation steps
# step 2: crop audio file to the data

# step 3: cut into different blocks and get rid of cyclic prefix




num_symbols = int(len(recording_without_chirp)/symbol_len)  # Number of symbols 

time_domain_datachunks = np.array(np.array_split(recording_without_chirp, num_symbols))[:, prefix_len:]

ofdm_datachunks = fft(time_domain_datachunks)  # Does the fft of all symbols individually 
ofdm_datachunks = ofdm_datachunks/channel_fft  # Divide each value by its corrosponding channel fft coefficient. 
data = ofdm_datachunks[:, lower_bin:upper_bin+1] # Selects the values from 1 to 511

data = data.flatten()

colors = np.where(source_data == 1+1j, "b", 
            np.where(source_data == -1+1j, "c", 
            np.where(source_data == -1-1j, "m", 
            np.where(source_data == 1-1j, "y", 
            "Error"))))

plt.scatter(data.real, data.imag, c=colors)
plt.show()






# step 4: take the DFT
# step 5: divide by channel coefficients determined in step 1
# step 6: choose complex values corresponding to information bits
# step 7: map each value to bits using QPSK decision regions
# step 8: decode recieved bits to information bits
# step 9: convert information bits to file using standardised preamble.