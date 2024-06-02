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




def generate_binary_preamble(file_name, file_size_bits):
    file_size_str = str(file_size_bits)

    # Create the byte sequence
    byte_sequence = bytearray()

    # Add two null bytes
    byte_sequence.extend(b'\x00\x00')

    # Add file name and two null bytes
    byte_sequence.extend(file_name.encode('utf-8'))
    byte_sequence.extend(b'\x00\x00')

    # Add the file size in bits followed by two null bytes
    byte_sequence.extend(file_size_str.encode('utf-8'))
    byte_sequence.extend(b'\x00\x00')

    preamble_bits = ''.join(format(byte, '08b') for byte in byte_sequence)
    # preamble_bits is a string. convert string to array
    binary_preamble = [int(bit) for bit in preamble_bits]
    # print(binary_preamble)
    return binary_preamble

bits = generate_binary_preamble(file_name, file_size_bits)

def extract_metadata(recovered_bitstream):
    byte_sequence = bytearray()

    # Convert the bitstream back to bytes (if this takes long then redesign this function)
    for i in range(0, len(recovered_bitstream), 8):
        byte = ''.join(str(bit) for bit in recovered_bitstream[i:i+8])
        byte_sequence.append(int(byte, 2))

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
    
    file_parts = file_name.split('.')
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


extract_metadata(bits)