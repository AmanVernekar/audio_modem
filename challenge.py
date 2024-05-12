import numpy as np  
import os 

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


file_num = 9  # File we are trying to decode (Must change these four values for each file)
raw_data_start = 216  # 216 for wav files and 232 for tiff files, can check this is correct as the second string printed 
                      # should be the ascii following the file size in the first string. 
file_size = 58584    # Must find this from the first printed string. 
file_type = 'wav'    # wav or bin so it doesn't save as a bin file, all of them have been tried so put this in before 
                      # you run the code so we don't make random files. (Just delete it if you do) 
file_string = file_to_binstr(file_num)   # Make a string of binary

def binary_to_ascii(binary_str):
    # Break the binary string into groups of 8 bits
    binary_chunks = [binary_str[i:i+8] for i in range(0, len(binary_str), 8)]

    # Convert each group of 8 bits to decimal and then to ASCII character
    ascii_chars = ''.join([chr(int(chunk, 2)) for chunk in binary_chunks])

    return ascii_chars


print(f"ASCII Result for file {file_num}:", binary_to_ascii(file_string[:1000]))
print(f"ASCII Result for file {file_num}:", binary_to_ascii(file_string[raw_data_start:1000]))
print() 


# define the path to the folder where you want to create the file 
folder_path = "C:/Users/sophi/OneDrive - University of Cambridge/Documents/audio_modem/Decodedfiles"
 
# define the file name and path 
file_name = f'binaryfile{file_num}.{file_type}' 
file_path = os.path.join(folder_path, file_name) 

file = open(file_path, 'wb')  # Make the file
def bitstring_to_bytes(s):
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')  # Function to turn the the string of bits into bytes
file_bytes = bitstring_to_bytes(file_string[raw_data_start:raw_data_start+(file_size*8)]) # The raw file data in bytes 
file.write(file_bytes)  # Write to our file
file.close() 
 