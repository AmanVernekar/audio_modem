import numpy as np

def compensate_for_phase_drift(remaining_data, phase_coefficients):
    """
    Compensate for phase drift in the remaining data.

    Parameters:
    remaining_data (numpy.ndarray): 2D array where each row corresponds to a block.
    phase_coefficients (numpy.ndarray): 1D array of phase coefficients.

    Returns:
    numpy.ndarray: Compensated remaining data.
    """
    # Ensure that remaining_data is a 2D array and phase_coefficients is a 1D array
    if remaining_data.ndim != 2:
        raise ValueError("remaining_data must be a 2D array")
    if phase_coefficients.ndim != 1:
        raise ValueError("phase_coefficients must be a 1D array")
    
    # Ensure the number of rows in remaining_data matches the length of phase_coefficients
    if remaining_data.shape[0] != phase_coefficients.shape[0]:
        raise ValueError("The number of rows in remaining_data must match the length of phase_coefficients")
    
    # Multiply each block by the corresponding phase coefficient
    compensated_remaining_data = remaining_data * phase_coefficients
    
    return compensated_remaining_data

# Example usage
remaining_data = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])

phase_coefficients = np.array([0.1, 0.2, 0.3])

compensated_remaining_data = compensate_for_phase_drift(remaining_data, phase_coefficients)
print(compensated_remaining_data)