import numpy as np
from bm4d import bm4d
from bm3d import bm3d

def bm4d_denoise(data, sigma):
    """
    Denoise a 3D complex tensor using BM4D algorithm.

    Parameters
    ----------
    data : ndarray
        Input tensor of shape (n, m, p).
    sigma : float
        Noise standard deviation.

    Returns
    -------
    ndarray
        Denoised tensor.
    """
    # Ensure the input data is a 3D tensor
    if data.ndim != 3 or np.any(data.shape[0] == 1):
        orginal_shape = data.shape
        denoised_data = bm3d(np.squeeze(data), sigma)
        return denoised_data.reshape(orginal_shape)

    # Perform BM4D denoising
    denoised_data = bm4d(data, sigma)

    return denoised_data
