import numpy as np
from scipy.signal import chirp
import matplotlib.pyplot as plt

chirp_duration = 5  # seconds
start_freq = 0.01
end_freq = 20
chirp_type = "linear"
prefix_len = 200

t = np.linspace(0, chirp_duration, 1000)  # time-values for chirp
chirp_sig = chirp(t, f0=start_freq, f1=end_freq, t1=chirp_duration, method=chirp_type)
chirp_sig = list(chirp_sig)  

chirp_prefix = chirp_sig[-prefix_len:]
chirp_suffix = chirp_sig[:prefix_len]
chirp_w_prefix_suffix = chirp_prefix + chirp_sig + chirp_suffix

x_values = np.linspace(0, len(chirp_w_prefix_suffix), len(chirp_w_prefix_suffix))

plt.figure(figsize=(10, 4))
plt.plot(x_values, chirp_w_prefix_suffix)
# plt.title("Chirp Signal with Prefix and Suffix")
# plt.xlabel("Sample Index")
# plt.ylabel("Amplitude")
plt.legend()
# plt.grid(True)
plt.show()
