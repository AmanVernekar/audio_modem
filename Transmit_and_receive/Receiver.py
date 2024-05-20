# step 1: 
#step 2: perform channel estimation and synchronisation steps
#step 2: crop audio file to the data
#step 3: cut into different blocks and get rid of cyclic prefix
#step 4: take the DFT
#step 5: divide by channel coefficients determined in step 1
#step 6: choose complex values corresponding to information bits
#step 7: map each value to bits using QPSK decision regions
#step 8: decode recieved bits to information bits
#step 9: convert information bits to file using standardised preamble.