import numpy as np
from numpy.fft import fft, ifft
import sounddevice as sd
import soundfile as sf
from scipy.signal import chirp

# STEP 1: encode file as binary data (e.g. LDPC)

prefix_len = 1024           # cyclic prefix length
datachunk_len = 2048        # length of data  
lower_freq = 1000           # lower frequency used for data
upper_freq = 11000          # upper frequency used for data
sample_rate = 44100         # sample rate 
repetition_factor = 5       # WHAT IS THIS?

# WHAT IS THE REPETITION?
coded_info_sequence = np.load("binary_data.npy")
rep_sequence = np.repeat(coded_info_sequence, repetition_factor)

# STEP 2: Modulate as complex symbols using QPSK
def qpsk_modulator(binary_sequence):
    # if binary_sequence has odd number of bits, add 0 at the end
    if len(binary_sequence) % 2 != 0:
        binary_sequence = np.append(binary_sequence, 0)
    
    # Initialize an empty array to store modulated symbols
    modulated_sequence = np.empty(len(binary_sequence) // 2, dtype=complex)
    
    # Mapping to complex symbols using QPSK   !! Do we care about energy of each symbol? !!
    for i in range(0, len(binary_sequence), 2):
        bit_pair = binary_sequence[i:i+2]
        # 00 => 1+j
        if np.array_equal(bit_pair, [0, 0]):
            modulated_sequence[i//2] = 1 + 1j
        # 01 => -1+j
        elif np.array_equal(bit_pair, [0, 1]):
            modulated_sequence[i//2] = -1 + 1j
        # 11 => -1-j
        elif np.array_equal(bit_pair, [1, 1]):
            modulated_sequence[i//2] = -1 - 1j
        # 11 => 1-j
        elif np.array_equal(bit_pair, [1, 0]):
            modulated_sequence[i//2] = 1 - 1j
    
    # print(f"QPSK Modulated sequence: {modulated_sequence}")
    return modulated_sequence

modulated_sequence = qpsk_modulator(rep_sequence) 
np.save("rep_mod_seq.npy", modulated_sequence)

# STEP 2.5: function for calculating bin values (optional)
def calculate_bins(sample_rate, lower_freq, upper_freq, ofdm_chunk_length):
    lower_bin = np.ceil((lower_freq / sample_rate) * ofdm_chunk_length).astype(int)  # round up
    upper_bin = np.floor((upper_freq / sample_rate) * ofdm_chunk_length).astype(int)  # round down

    print(f"""
    for the parameters: sample rate = {sample_rate}Hz
                        information bandlimited to {lower_freq} - {upper_freq}Hz
                        OFDM symbol length = {ofdm_chunk_length}
                lower bin is {lower_bin}
                upper bin is {upper_bin}
    """)
    return lower_bin, upper_bin

lower_bin, upper_bin = calculate_bins(sample_rate, lower_freq, upper_freq, datachunk_len)

# STEP 3: insert QPSK complex values into as many OFDM datachunks as required 
def create_ofdm_datachunks(modulated_sequence, chunk_length, lower_bin, upper_bin):
    #  calculate number of information bins
    num_information_bins = (upper_bin - lower_bin) + 1

    # append with 0s if not modulated_sequence is not multiple of num_information_bins
    num_zeros = num_information_bins - (len(modulated_sequence) % num_information_bins)

    if num_zeros != 0:
        zero_block = np.zeros(num_zeros, dtype=complex)
        modulated_sequence = np.append(modulated_sequence, zero_block)

    # create new array containing modulated_sequence, where each row corresponds to an OFDM data chunk
    separated_mod_sequence = np.reshape(modulated_sequence, (-1, num_information_bins))  

    # create a complex array of ofdm data chunks, where each symbol is an array filled with 0s of length chunk_length
    num_of_symbols = separated_mod_sequence.shape[0]
    ofdm_datachunk_array = np.zeros((num_of_symbols, chunk_length), dtype=complex)  # change this so not zeros

    # insert information in OFDM blocks: 
    ofdm_datachunk_array[:, lower_bin:upper_bin+1] = separated_mod_sequence  # populates first half of block
    ofdm_datachunk_array[:, chunk_length-upper_bin:(chunk_length-lower_bin)+1] = np.fliplr(np.conjugate(separated_mod_sequence))  # second half of block
 
    return ofdm_datachunk_array  # returns array of OFDM blocks

ofdm_datachunks = create_ofdm_datachunks(modulated_sequence, datachunk_len, lower_bin, upper_bin)

# STEP 4: IDFT each OFDM symbol
time_domain_datachunks = ifft(ofdm_datachunks, axis=1)  # applies ifft to each row
time_domain_datachunks = time_domain_datachunks.real  # takes real part of ifft

# STEP 5: add cyclic prefix to each part
def add_cyclic_prefix(ofdm_datachunks, prefix_length):
    block_prefixes = ofdm_datachunks[:, -prefix_length:]
    ofdm_symbols = np.concatenate((block_prefixes, ofdm_datachunks), axis=1)
    return ofdm_symbols

ofdm_symbols = add_cyclic_prefix(time_domain_datachunks, prefix_len)

# STEP 6: flatten all time domain blocks into one array
concatenated_blocks = ofdm_symbols.flatten()

# STEP 7:convert to audio file to transmit across channel. add chirp beforehand etc.
def convert_data_to_audio(data, sampling_rate):
    # normalise to between -1 and 1:
    max_absolute_val = np.max(np.abs(data))
    waveform = data / max_absolute_val

    # play the waveform
    # sd.play(waveform, sampling_rate)
    # sd.wait()
    # np.save("rep_waveform.npy", waveform)
    return waveform 

# Chirp Parameters
chirp_duration = 5  # seconds
start_sig = [0]*sample_rate  # 1 second silence
start_freq = 0.01
end_freq = 22050
chirp_type = "linear" 
cyclic_prefix = 512

t = np.linspace(0, chirp_duration, int(chirp_duration*sample_rate))  # time-values for chirp
chirp_sig = chirp(t, f0=start_freq, f1=end_freq, t1=chirp_duration, method=chirp_type)
chirp_sig = list(chirp_sig)  

chirp_prefix = chirp_sig[-cyclic_prefix:]
chirp_suffix = chirp_sig[:cyclic_prefix]
chirp_w_prefix_suffix = chirp_prefix + chirp_sig + chirp_suffix

rep_waveform = convert_data_to_audio(concatenated_blocks, sample_rate)
overall_sig = start_sig + chirp_w_prefix_suffix + list(rep_waveform)

# Play the audio data
sd.play(overall_sig, sample_rate)
sd.wait()  # Wait until the sound has finished playing

output_file = 'rep_audio_to_test_with.wav'
sf.write(output_file, overall_sig, sample_rate)

print(f"Samples of data: {len(concatenated_blocks)}")
