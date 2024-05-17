import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import correlate, chirp
from numpy.fft import fft, ifft

chirp_duration = 2
start_freq = 1000
end_freq = 11000
chirp_type = "linear" 
sample_rate = 44100

t_chirp = np.linspace(0, chirp_duration, int(sample_rate * chirp_duration), endpoint=False)

chirp_sig = chirp(t_chirp, f0=start_freq, f1=end_freq, t1=chirp_duration, method=chirp_type)
# chirp_sig = list(chirp_sig)

a = chirp_sig * chirp_sig
plt.plot(a)
plt.show()

plt.plot(np.abs(fft(a)))
plt.show()