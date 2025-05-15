# %%
import numpy as np
import matplotlib.pyplot as plt

from denoise.denoise_bm4d import bm4d_denoise
from denoise.patch_denoise import locally_low_rank_tucker, tucker, tucker_to_tensor


from phantom import phantom_multicontrast


def plot_images(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 15))
    for ax, img, title in zip(axes, images, titles):
        # ax.imshow(img, cmap='gray')
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.show()

# %%

data = phantom_multicontrast(32)

# %%

noise_level = 0.1
data_noisy = data + noise_level * (np.random.randn(*data.shape) + 1j * np.random.randn(*data.shape))

bm4d_denoised = np.squeeze(bm4d_denoise(np.abs(data_noisy[np.newaxis, :, :, 0]), noise_level))


# %%


plot_images(
    [np.abs(data[:, :, 0]), np.abs(data_noisy[:, :, 0]), bm4d_denoised],
    ["Original", "Noisy", "BM4D Denoised"]
)
# %%

tucker_denoised = locally_low_rank_tucker(np.abs(data_noisy), 3.0, tau=0.1, patch_transform="fft", lambda2d=1)

# %%

plot_images(
    [np.abs(data[:, :, 0]), np.abs(data_noisy[:, :, 0]), np.abs(tucker_denoised[:, :, 0])],
    ["Original", "Noisy", "Tucker Denoised"]
)

# %%
