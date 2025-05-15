import numpy as np

from phantom import phantom_multicontrast
from denoise.patch_denoise import locally_low_rank_tucker

data = phantom_multicontrast(32)
noise_level = 0.1
data_noisy = data + noise_level * (np.random.randn(*data.shape) + 1j * np.random.randn(*data.shape))

locally_low_rank_tucker(data_noisy, 3, tau=0.5/noise_level, patch_transform="wavelet", lambda2d=0.1)
