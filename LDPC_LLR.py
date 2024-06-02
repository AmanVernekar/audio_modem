from ldpc_jossy.py import ldpc
import numpy as np 
from math import sqrt

from Transmit_and_receive import parameters 

# Need to compute LLRs from symbols. 

# Each symbol uses 648 bins 
# This means we have 648 complex values, we need to compute 648 x 2 = 1296 LLRs, one for each bit 
# Then we have 1296 LLRs for one block of code bits. 
# These LLRs will be put through a decoder which outputs belief values which are used with decision boundaries to creates 1s and 0s
# The first half is taken as it is a systematic code. (IDK WHAT THE SECOND HALF IS????)


# Computing LLRs for a given noise variance 
# We are given a bunch of complex values Y' = X + N_k which is the noisy modulated data (divided by channel coefs, what we plotted)

# First define a function that takes one complex value, its channel coefficient, noise variance, and gain A. 
# Which computes the two LLRs. 

def two_bit_LLR(y, c, sigma, A): 
    c_conj = c.conjugate()
    L_1 = (A*c*c_conj*sqrt(2)*y.imag) / (sigma*sigma)
    L_2 = (A*c*c_conj*sqrt(2)*y.real) / (sigma*sigma)

    return L_1, L_2 

# Now we want to have 648 bin values and create a np array of 1296 LLRs 

def random_complex_array():
    # Define the four complex values
    complex_values = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])

    # Generate an array of 648 random choices from the complex values
    random_complex_array = np.random.choice(complex_values, 648)

    # Define the standard deviation of the Gaussian noise
    noise_std = 0.1  # You can adjust the standard deviation as needed

    # Generate Gaussian noise for the real and imaginary parts
    real_noise = np.random.normal(0, noise_std, random_complex_array.shape)
    imag_noise = np.random.normal(0, noise_std, random_complex_array.shape)

    # Add the noise to the complex array
    noisy_complex_array = random_complex_array + real_noise + 1j * imag_noise

    return noisy_complex_array

rand_complex_array = random_complex_array()
channel_coefs = []
sigma_squared = 1
A = 1
LLR_list = []

for i in range(len(rand_complex_array)): 
    L_1 = two_bit_LLR(rand_complex_array[i], channel_coefs[i], sigma_squared, A)[0]
    LLR_list.append(L_1)
    L_2 = two_bit_LLR(rand_complex_array[i], channel_coefs[i], sigma_squared, A)[1]
    LLR_list.append(L_2)

# Now we have the LLR values to put into the decoder so we just want to figure out how to find sigma and A. 

# To find sigma we want to get the FFT of the recorded symbol and for each frequency bin find N = Y - c_kX
# Where c_k are the channel coefficients

noise = []
ofdm_known_symbol = [] # This is the fft the known ofdm symbol from recording
ofdm_symbol_sent = [] # This is the fft of the send ofdm symbol i.e X 

for i in range(len(ofdm_known_symbol)): 
    noise_complex = ofdm_known_symbol[i] - channel_coefs[i]*ofdm_symbol_sent[i]
    n1 = (noise_complex.real)*(noise_complex.real)
    n2 = (noise_complex.imag)*(noise_complex.imag)
    noise.append(n1)
    noise.append(n2)

sigma_squared = sum(noise) / len(noise)