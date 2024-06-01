# we have file we want to transmit saved in \files_to_transmit
# convert this to binary
# add standard preamble

# For any files:
# ‘\0\0FileName.type\0\0numBits\0\0’

# In UTF8

import numpy as np

file_path = 'Transmit_and_receive/files_to_transmit/example.txt'

# Extract file name
file_name = file_path.split('/')[-1]  # Extracts the file name from the file path

# File as binary array
with open(file_path, 'rb') as file:
    # Read the file content into a binary string
    file_content = file.read()
    file_size_bytes = len(file_content)
    # Convert the binary content to a numpy array
    file_as_binary_array = np.unpackbits(np.frombuffer(file_content, dtype=np.uint8))

# Calculate file size in bits
file_size_bits = file_size_bytes * 8

# # Print file information
# print("File Name:", file_name)
# print("File Size (bytes):", file_size_bytes)
# print("File Size (bits):", file_size_bits)
# print("Binary array:", file_as_binary_array)

# Create binary preamble
# preamble = "\\0\\0{}\\0\\0{}\\0\\0".format(file_name, file_size_bits)
# print("Preamble:", preamble)
# preamble_utf8 = preamble.encode('utf-8')
# print("Preamble (UTF-8):", preamble_utf8)
# Convert file size to string
file_size_str = str(file_size_bits)

# Create the byte sequence
byte_sequence = bytearray()

# Add two null bytes
byte_sequence.extend(b'\x00\x00')

# Add file name followed by '.type' and two null bytes
byte_sequence.extend(file_name.encode('utf-8'))
byte_sequence.extend(b'\x00\x00')

# Add the file size in bits followed by two null bytes
byte_sequence.extend(file_size_str.encode('utf-8'))
byte_sequence.extend(b'\x00\x00')

raw_bits = ''.join(format(byte, '08b') for byte in byte_sequence)
print("Preamble raw bits:", raw_bits)

def bits_to_utf8(bits):
    # Split bits into bytes
    bytes_list = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]
    # Convert bytes to byte objects
    byte_objects = bytes(bytes_list)
    # Decode UTF-8
    text = byte_objects.decode('utf-8')
    return text

decoded_preamble = bits_to_utf8(raw_bits)
print(decoded_preamble)

# Decode the binary content to a string and print the content of the text file
text_content = file_content.decode('utf-8')  # Adjust the encoding if necessary
print("\nText content:")
print(text_content)

