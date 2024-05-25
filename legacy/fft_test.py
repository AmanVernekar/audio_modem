import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import correlate, chirp, find_peaks
from numpy.fft import fft, ifft

from scipy.io.wavfile import read
a = read("Testing_outputs/yo1.wav")
a = np.array(a[1],dtype=float)

chirp_duration = 2
start_freq = 0.1
end_freq = 22050
chirp_type = "linear" 
sample_rate = 44100

t_chirp = np.linspace(0, chirp_duration, int(sample_rate * chirp_duration), endpoint=False)

chirp_sig = chirp(t_chirp, f0=start_freq, f1=end_freq, t1=chirp_duration, method=chirp_type)
# chirp_sig = list(chirp_sig)

matched_filter_output = correlate(a, chirp_sig, mode='same')

peaks, properties = find_peaks(matched_filter_output, distance=40000, height=5e5)
heights = properties['peak_heights']
print(peaks)
print(heights)

plt.plot(peaks, heights)
plt.plot(matched_filter_output)
# plt.plot(find_peaks(matched_filter_output)[0])
plt.show()

