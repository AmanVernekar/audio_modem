import numpy as np  

def file_to_binstr(file_num):

    channel = np.genfromtxt('dataset/channel.csv')  # Imports the channel file as an array. 

    block_len = 1024
    prefix_len = 32
    symbol_len = block_len + prefix_len
    channel_len = len(channel)

    file = np.genfromtxt(f"dataset/file{file_num}.csv") # Import the file1 data into an array. 
    num_symbols = int(len(file)/symbol_len)  # Number of symbols 
    file = np.array(np.array_split(file, num_symbols))[:, 32:]   # Splits into symbols sent by the channel (length 1056) and removes first 32. 
    #print(file.shape)

    file = np.fft.fft(file)  # Does the fft of all symbols individually 
    channel = np.fft.fft(np.concatenate((channel, [0]*(block_len - channel_len))))   # Zero pads the end of the channel pulse and takes fft. 
    file = file/channel  # Divide each value by its corrosponding channel fft coefficient. 
    file = file[:, 1:512] # Selects the values from 1 to 511
    #print(file.shape)
    #print(f1[0])

    # Desicion rules
    condition_00 = (file.real >= 0) & (file.imag >= 0)
    condition_01 = (file.real < 0) & (file.imag >= 0)
    condition_11 = (file.real < 0) & (file.imag < 0)
    condition_10 = (file.real >= 0) & (file.imag < 0)


    # Apply the decision rules using np.where
    file = np.where(condition_00, "00", 
                np.where(condition_01, "01", 
                np.where(condition_11, "11", 
                np.where(condition_10, "10", 
                "Error"))))

    file = file.ravel() # makes the array 1D
    file_string = "".join(file) #creates string from 1D array
    return file_string


file_num = 1   # File we are trying to decode
f1_file_string = file_to_binstr(file_num)   # Make a string of binary

def binary_to_ascii(binary_str):
    # Break the binary string into groups of 8 bits
    binary_chunks = [binary_str[i:i+8] for i in range(0, len(binary_str), 8)]

    # Convert each group of 8 bits to decimal and then to ASCII character
    ascii_chars = ''.join([chr(int(chunk, 2)) for chunk in binary_chunks])

    return ascii_chars


print(f"ASCII Result for file {file_num}:", binary_to_ascii(f1_file_string[:10000]))
print(f"ASCII Result for file {file_num}:", binary_to_ascii(f1_file_string[232:1000]))
print() 

file = open(f'binaryfile{file_num}.bin', 'wb')  # Make a binary file
def bitstring_to_bytes(s):
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')  # Function to turn the the string of bits into bytes
file_bytes = bitstring_to_bytes(f1_file_string[232:232+(174068*8)]) # The raw file data in bytes 
file.write(file_bytes)  # Write to our binary file
file.close() 