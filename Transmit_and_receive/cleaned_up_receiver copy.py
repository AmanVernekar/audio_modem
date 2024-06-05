import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import chirp, correlate
from math import sqrt
from ldpc_jossy.py import ldpc

import parameters
import our_chirp

# length of the data in the OFDM symbol
datachunk_len = parameters.datachunk_len
prefix_len = parameters.prefix_len                   # length of cyclic prefix
symbol_len = parameters.symbol_len                   # total length of symbol
sample_rate = parameters.sample_rate                 # samples per second
# duration of recording in seconds
rec_duration = parameters.rec_duration
# duration of chirp in seconds
chirp_duration = parameters.chirp_duration
chirp_start_freq = parameters.chirp_start_freq       # chirp start freq
chirp_end_freq = parameters.chirp_end_freq           # chirp end freq
chirp_type = parameters.chirp_type                   # chirp type
# number of samples of data (HOW IS THIS FOUND)
recording_data_len = parameters.recording_data_len
lower_bin = parameters.lower_bin
upper_bin = parameters.upper_bin
num_data_bins = upper_bin-lower_bin+1
num_known_symbols = 1
chirp_samples = int(sample_rate * chirp_duration)
known_datachunk = parameters.known_datachunk
known_datachunk = known_datachunk.reshape(1, 4096)

# STEP 1: Generate transmitted chirp and record signal
chirp_sig = our_chirp.chirp_sig

do_real_recording = True

# Determines if we record in real life or get file which is already recorded
if do_real_recording:
    # Using real recording
    recording = sd.rec(sample_rate*rec_duration,
                       samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()

    recording = recording.flatten()  # Flatten to 1D array if necessary
    np.save(f"Data_files/example_file_recording_to_test_with.npy", recording)

else:
    # Using saved recording
    recording = np.load(f"Data_files/example_file_recording_to_test_with.npy")

# STEP 2: Initial Synchronisation

# The matched_filter_output has been changed to full as this is the default method and might be easier to work with
# this means that the max index detected is now at the end of the chirp
matched_filter_output = correlate(recording, chirp_sig, mode='full')
matched_filter_first_half = matched_filter_output[:int(
    len(matched_filter_output)/2)]

detected_index = np.argmax(matched_filter_first_half)
print(detected_index)
print(f"The index of the matched filter output is {detected_index}")


# Re-sync off for now
do_cyclic_resynch = False

if do_cyclic_resynch:
    # Use matched filter to take out the chirp from the recording
    chirp_fft = fft(chirp_sig)
    detected_chirp = recording[detected_index-chirp_samples:detected_index]
    detected_fft = fft(detected_chirp)
    channel_fft = detected_fft/chirp_fft
    channel_impulse = ifft(channel_fft)

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
        # print(impulse_start)
        if impulse_start > len(channel_impulse) / 2:
            impulse_start = impulse_start - len(channel_impulse)
        # print(impulse_start)
        return impulse_start

    impulse_shift = impulse_start_max(channel_impulse)
else:
    impulse_shift = 0

# Process through which we calculate optimal shift with error rates from known OFDM symbols
# Off for now

optimisation_resynch = False


def estimate_channel_from_known_ofdm():
    channel_fft = ofdm_datachunks[0]/known_datachunk[0]
    return channel_fft


if optimisation_resynch:
    shifts = range(-100, 100)
    total_errors = np.zeros((len(shifts)))

    # -------------------------------------------------------------------------------------------------------------
    # Please look at this bit I don't get it
    source_mod_seq = np.load(
        f"Data_files/mod_seq_example_file.npy")[num_known_symbols*num_data_bins:]

    sent_signal = np.load(f'Data_files/example_file_overall_sent.npy')
    data_start_sent_signal = sample_rate + (prefix_len*2) + (chirp_samples)
    end_start_sent_signal = (prefix_len*2) + (chirp_samples)
    sent_without_chirp = sent_signal[data_start_sent_signal: -
                                     end_start_sent_signal]
    print("sent data length", len(sent_without_chirp))
    num_symbols = int(len(sent_without_chirp)/symbol_len)
    print("num of symbols: ", num_symbols)
    sent_datachunks = np.array(np.array_split(
        sent_without_chirp, num_symbols))[:, prefix_len:]

    colors = np.where(source_mod_seq == (1+1j), "b",
                      np.where(source_mod_seq == (-1+1j), "c",
                               np.where(source_mod_seq == (-1-1j), "m",
                                        np.where(source_mod_seq == (1-1j), "y",
                                                 "Error"))))

    for g, shift in enumerate(shifts):

        data_start_index = detected_index+shift+prefix_len
        recording_without_chirp = recording[data_start_index:
                                            data_start_index+recording_data_len]

        # STEP 5: cut into different blocks and get rid of cyclic prefix
        num_symbols = int(len(recording_without_chirp) /
                          symbol_len)  # Number of symbols
        if g == 0:
            print("num_symbols_calc: ", num_symbols)
        time_domain_datachunks = np.array(np.array_split(
            recording_without_chirp, num_symbols))[:, prefix_len:]
        # Does the fft of all symbols individually
        ofdm_datachunks = fft(time_domain_datachunks)

        channel_estimate_from_first_symbol = estimate_channel_from_known_ofdm()

        # Divide each value by its corrosponding channel fft coefficient.
        ofdm_datachunks = ofdm_datachunks[num_known_symbols:]/channel_estimate_from_first_symbol
        # Selects the values from 1 to 511
        data = ofdm_datachunks[:, lower_bin:upper_bin+1]

        total_error = 0

        _data = data[:num_known_symbols].flatten()
        _source_mod_seq = source_mod_seq[:num_known_symbols * num_data_bins]
        for w, value in enumerate(_data):
            sent = _source_mod_seq[w]
            if value.real/sent.real < 0 or value.imag/sent.imag < 0:
                total_error = total_error + 1

        total_errors[g] = total_error*10/len(_data)

    plt.plot(shifts, total_errors)
    plt.axvline(shifts[np.argmin(total_errors)])
    plt.ylabel("Bit error percentage (%)")
    plt.xlabel("Index")
    # plt.show()

    best_shift = shifts[np.argmin(total_errors)]
    print(best_shift)
    print(np.min(total_errors))

else:
    best_shift = 0


# STEP: Get complex values from the data using our best synchronisation estimate:
# I haven't looked at this bit
# -------------------------------------------------------------------------------------------------------------

# Refinding the data from the best shift.
data_start_index = detected_index+best_shift+prefix_len
recording_without_chirp = recording[data_start_index:
                                    data_start_index+recording_data_len]

num_symbols = int(len(recording_without_chirp)/symbol_len)  # Number of symbols
time_domain_datachunks = np.array(np.array_split(
    recording_without_chirp, num_symbols))[:, prefix_len:]
# Does the fft of all symbols individually
ofdm_datachunks = fft(time_domain_datachunks)

channel_estimate_from_first_symbol = estimate_channel_from_known_ofdm()

# Divide each value by its corrosponding channel fft coefficient.
ofdm_datachunks = ofdm_datachunks[num_known_symbols:]/channel_estimate_from_first_symbol
# Selects the values from 1 to 511
data_complex = ofdm_datachunks[:, lower_bin:upper_bin+1]

num_unknown_symbols = num_symbols - num_known_symbols

# -------------------------------------------------------------------------------------------------------------

# Move all of the LDPC stuff below in here

def complex_data_hard_decision_to_binary(data_complex, num_unknown_symbols, num_data_bins):
    """Uses hard decision boundaries to map complex data to binary"""
    
    # This uses our current code. However, surely data_bin should have length len(data_complex)? maybe not

    data_bin = np.zeros((num_unknown_symbols, num_data_bins, 2), dtype=int)

    data_bin[(data_complex.real > 0) & (
        data_complex.imag > 0)] = [0, 0]  # [0, 0]
    data_bin[(data_complex.real < 0) & (
        data_complex.imag > 0)] = [0, 1]  # [0, 1]
    data_bin[(data_complex.real < 0) & (
        data_complex.imag < 0)] = [1, 1]  # [1, 1]
    data_bin[(data_complex.real > 0) & (
        data_complex.imag < 0)] = [1, 0]  # [1, 0]

    return data_bin

def do_ldpc_decoding(complex_data):
    """Uses LDPC to go from complex data from 1 received OFDM symbol to the information data"""
    hard_binary_data, soft_binary_data = (0, 0)  # placeholder
    return hard_binary_data, soft_binary_data


def decode_without_ldpc():
    """Returns decoded binary array"""

    hard_decision_binary_data = complex_data_hard_decision_to_binary(data_complex, num_unknown_symbols, num_data_bins)
    # Reshape the array to (105, 1296)
    hard_decision_binary_data = hard_decision_binary_data.reshape(num_unknown_symbols, num_data_bins*2)

    first_half_systematic_data = hard_decision_binary_data[:, :num_data_bins]
    # print("shape of binary data: ", first_half_systematic_data.shape)

    flattened_first_halfs = first_half_systematic_data.flatten()
    flattened_first_halfs = list(flattened_first_halfs)

    print(f"Without LDPC as UTF8: {binary_to_utf8(flattened_first_halfs)}")

    return flattened_first_halfs

def decode_ldpc_hard_decision():
    """Returns decoded binary array"""
    return None

def decode_ldpc_with_real_LLRs(): 
    """Returns decoded binary array"""
    return None

hard_decision_binary_data = complex_data_hard_decision_to_binary(data_complex, num_unknown_symbols, num_data_bins)
# Reshape the array to (105, 1296)
hard_decision_binary_data = hard_decision_binary_data.reshape(num_unknown_symbols, num_data_bins*2)


first_half_systematic_data = hard_decision_binary_data[:, :num_data_bins]
# print("shape of binary data: ", first_half_systematic_data.shape)

flattened_first_halfs = first_half_systematic_data.flatten()
# flattened_first_halfs = list(flattened_first_halfs)


def binary_to_utf8(binary_list):
    # Join the list of integers into a single string
    binary_str = ''.join(str(bit) for bit in binary_list)

    # Split the binary string into 8-bit chunks (bytes)
    bytes_list = [binary_str[i:i+8] for i in range(0, len(binary_str), 8)]

    # Convert each byte to its corresponding UTF-8 character
    utf8_chars = [chr(int(byte, 2)) for byte in bytes_list]

    # Join the UTF-8 characters to form the final string
    utf8_string = ''.join(utf8_chars)

    return utf8_string


# Not currently in use:


def extract_metadata(recovered_bitstream):
    byte_sequence = bytearray()

    # Convert the bitstream back to bytes (if this takes long then redesign this function)
    for i in range(0, len(recovered_bitstream), 8):
        byte = ''.join(str(bit) for bit in recovered_bitstream[i:i+8])
        byte_sequence.append(int(byte, 2))

    # # convert byte sequence to ascii
    # byte_as_ascii = ''.join(chr(byte) for byte in byte_sequence)
    # print(byte_as_ascii)

    # Extract file name and type
    null_byte_count = 0
    file_name_and_type = ""
    for byte in byte_sequence:
        if byte == 0:
            null_byte_count += 1
            if null_byte_count == 3:
                break
        else:
            file_name_and_type += chr(byte)

    file_parts = file_name_and_type.split('.')
    file_name = file_parts[0]  # The part before the dot
    file_type = file_parts[1]  # The part after the dot

    # Extract file size in bits
    file_size_bits = ""
    for byte in byte_sequence[len(file_name_and_type) + 4:]:
        if byte == 0:
            break
        file_size_bits += chr(byte)

    # Convert file size back to integer
    file_size_bits = int(file_size_bits)

    return file_name, file_type, file_size_bits

# extract_metadata(first_half_systematic_data)


ldpc_z = parameters.ldpc_z
c = ldpc.code('802.16', '1/2', ldpc_z)


fake_LLR_multiply = 5

fake_LLR_from_bin = fake_LLR_multiply * (0.5 - hard_decision_binary_data)

app, it = c.decode(fake_LLR_from_bin[1])
app = app[:648]
x = np.load(f"Data_files/example_file_data_extended_zeros.npy")


compare1 = first_half_systematic_data[1]
compare2 = x[648:648*2]

app = np.where(app < 0, 1, 0)
compare3 = app

print(f"LDPC with hard decisions as UTF8: {binary_to_utf8(app)}")


def error(compare1, compare2, test):
    assert len(compare1) == len(compare2)
    wrong = np.count_nonzero(compare1-compare2)
    print("# of bit errors: ", wrong)
    print(test, " : ", (wrong / len(compare1))*100, "%")


error(compare1, compare2, '1 against 2')
error(compare2, compare3, '2 against 3')


def LLRs(complex_vals, c_k, sigma_square, A):
    LLR_list = []
    for i in range(len(complex_vals)):
        # c_conj = c_k[i].conjugate()
        c_squared = (np.abs(c_k[i]))**2
        L_1 = (A*c_squared*complex_vals[i].imag) / (sigma_square)
        LLR_list.append(L_1)
        L_2 = (A*c_squared*complex_vals[i].real) / (sigma_square)
        LLR_list.append(L_2)

    return LLR_list


def decode_data(LLRs, chunks_num):
    LLRs_split = np.array(np.array_split(LLRs, chunks_num))

    decoded_list = []
    for i in range(chunks_num):
        decoded_chunk, it = c.decode(LLRs_split[i])
        decoded_list.append(decoded_chunk)

    decoded_data = np.concatenate(decoded_list)
    decoded_data = np.where(decoded_data < 0, 1, 0)

    decoded_data_split = np.array(
        np.array_split(decoded_data, chunks_num))[:, : 648]
    decoded_raw_data = np.concatenate(decoded_data_split)

    return decoded_raw_data


c_k = channel_estimate_from_first_symbol[lower_bin:upper_bin+1]


def average_magnitude(complex_array):
    # Calculate the magnitudes of the complex numbers
    magnitudes = np.abs(complex_array)

    # Calculate the average of the magnitudes
    average_mag = np.mean(magnitudes)

    return average_mag


A = average_magnitude(data_complex[0])
print("A: ", A)

sigma_vals = [1]
complex_vals = data_complex.flatten()

for i in sigma_vals:
    LLRs_block_1 = LLRs(complex_vals, c_k, i, A)
    decoded_raw_data = decode_data(LLRs_block_1, chunks_num=1)
    compare4 = decoded_raw_data
    error(compare2, compare4, '2 against 4')
