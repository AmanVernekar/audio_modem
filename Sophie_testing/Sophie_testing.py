import numpy as np
from scipy.signal import chirp
import os 
import soundfile as sf

# Tone Parameters
tone_duration = 1.0  # seconds
tone_frequency = 800.0  # Hz 
sample_rate = 44100  # samples per second

# Generate a tone to check our chirp detection is correct. 
t_tone = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)
tone = 0.5 * np.sin(2 * np.pi * tone_frequency * t_tone)  # Try wihtout 0.5

# Chirp Parameters
chirp_t = 2
start_sig = [0]*sample_rate
half_s_samples = int(0.5*sample_rate) #generates half a second of samples 
gap = [0]*half_s_samples
start_freq = 100
end_freq = 18000
chirp_type = "logarithmic" 

t = np.linspace(0, chirp_t, chirp_t*sample_rate)
chirp_sig = chirp(t, f0=start_freq, f1=end_freq, t1=chirp_t, method=chirp_type)
chirp_sig = list(chirp_sig) 
chirp_prefix = chirp_sig[-15000:]
chirp_w_prefix = chirp_prefix + chirp_sig
overall_sig = start_sig + chirp_w_prefix + gap + list(tone)


# define the path to the folder where you want to create the file 
folder_path = "C:/Users/sophi/OneDrive - University of Cambridge/Documents/audio_modem/Sophie_testing"

# define the file name and path 
file_name = f'{chirp_type}_f0_{start_freq}_f1_{end_freq}_time_{chirp_t}.wav' 
file_path = os.path.join(folder_path, file_name) 
print(file_path)

#file = open(file_path, 'w')  # Make the file'
sf.write(file_path, overall_sig, sample_rate)
#file.write(str(overall_sig))  # Write to our file
#file.close() 