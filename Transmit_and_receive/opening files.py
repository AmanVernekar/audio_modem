#-----------------------------------------------------
# STEP 6: Convert information bits to file using standardised preamble.
#-----------------------------------------------------

# given information bits and standard preamble, save (and open) the file

# For any files:
# ‘\0\0FileName.type\0\0numBits\0\0’

# In UTF8

# This header is prepended as raw bits to the raw bits of the file. The full bit stream is then encoded using LDPC as outlined below
