import numpy as np
import parameters

known_datachunk = parameters.known_datachunk
known_datachunk = known_datachunk.reshape(1, 4096)
lower_bin = parameters.lower_bin
upper_bin = parameters.upper_bin

# Define the possible complex numbers
complex_numbers = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])

# Generate an array of shape (1, 648) filled with random choices from the complex_numbers array
separated_mod_sequence = np.random.choice(complex_numbers, size=(1, 648))


phases = np.where(separated_mod_sequence == (1+1j), 0, 
            np.where(separated_mod_sequence == (-1+1j), np.pi/2, 
            np.where(separated_mod_sequence == (-1-1j), (2 * np.pi)/2, 
            np.where(separated_mod_sequence == (1-1j), (3 * np.pi)/2, 
            np.nan))))
ofdm_datachunk_sub = known_datachunk[:, lower_bin:upper_bin+1]
ofdm_sub_rotated = ofdm_datachunk_sub * np.exp(1j * phases)

# print(ofdm_datachunk_sub[0][:20])
# print(separated_mod_sequence[0][:20])
# print(ofdm_sub_rotated[0][:20])

data_complex = ofdm_sub_rotated
# print(data_complex.shape)


known_datachunk_data_bins = known_datachunk[0][lower_bin:upper_bin+1]
phases = np.where(np.isclose(known_datachunk_data_bins, (1+1j)), 0, 
    np.where(np.isclose(known_datachunk_data_bins, (-1+1j)), np.pi/2, 
    np.where(np.isclose(known_datachunk_data_bins, (-1-1j)), (2 * np.pi)/2, 
    np.where(np.isclose(known_datachunk_data_bins, (1-1j)), (3 * np.pi)/2, 
    np.nan))))
data_complex = data_complex / np.exp(1j * phases)

print(separated_mod_sequence)


# print(data_complex == separated_mod_sequence)
comparison_result = np.isclose(data_complex, separated_mod_sequence, atol=1e-8)
print(comparison_result)






