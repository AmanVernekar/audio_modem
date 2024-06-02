import parameters
import numpy as np
from scipy.signal import chirp

chirp_duration = parameters.chirp_duration  # seconds
start_sig = [0]*parameters.sample_rate  # 1 second silence
start_freq = parameters.chirp_start_freq
end_freq = parameters.chirp_end_freq
chirp_type = parameters.chirp_type
cyclic_prefix = parameters.prefix_len
chirp_reduction = parameters.chirp_reduction

t = np.linspace(0, chirp_duration, int(chirp_duration*parameters.sample_rate))  # time-values for chirp
chirp_sig = chirp(t, f0=start_freq, f1=end_freq, t1=chirp_duration, method=chirp_type)
chirp_sig = list(chirp_sig)  

chirp_prefix = chirp_sig[-parameters.prefix_len:]
chirp_suffix = chirp_sig[:parameters.prefix_len]
chirp_w_prefix_suffix = chirp_prefix + chirp_sig + chirp_suffix

chirp_w_prefix_suffix = np.array(chirp_w_prefix_suffix)
chirp_w_prefix_suffix = chirp_reduction * chirp_w_prefix_suffix
chirp_w_prefix_suffix = list(chirp_w_prefix_suffix)
