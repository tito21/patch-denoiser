# %%
import numpy as np
import matplotlib.pyplot as plt
from numba import config

# config.DISABLE_JIT = True  # Disable JIT compilation for debugging


from denoise.denoise_bm4d import bm4d_denoise
from denoise.patch_denoise import locally_low_rank_tucker
from denoise.patch2self import patch2self

from phantom import phantom_multicontrast, transform_channels


def plot_images(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 15))
    for ax, img, title in zip(axes, images, titles):
        # ax.imshow(img, cmap='gray')
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.show()

def rmse(img1, img2):
    """Compute the root mean square error between two images."""
    return np.sqrt(np.mean((np.abs(img1) - np.abs(img2)) ** 2))

# %%

data = phantom_multicontrast(64, n_contrast=15)
data = transform_channels(data)
# data = np.load("slice_0.npy")
data = np.stack([data] * 1, axis=0)


plot_images([np.abs(data[0, :, :, i]) for i in range(data.shape[-1])], [f"Contrast {i}" for i in range(data.shape[-1])])


# %%

noise_level = 0.025
# noise_level = 0.0
data_noisy = data + noise_level * (np.random.randn(*data.shape) + 1j * np.random.randn(*data.shape))
# data_noisy = data
bm4d_denoised = np.stack([bm4d_denoise(np.abs(data_noisy[:, :, :, i]), noise_level) for i in range(data.shape[-1])], axis=-1)
# %%

slice_idx = 0  # Change this to visualize different slices

for i in range(data.shape[-1]):
    plot_images(
        [np.abs(data[slice_idx, :, :, i]), np.abs(data_noisy[slice_idx, :, :, i]), np.abs(bm4d_denoised[slice_idx, :, :, i]), np.abs(data_noisy[slice_idx, :, :, i] - bm4d_denoised[slice_idx, :, :, i])],
        ["Original", f"Noisy {rmse(np.abs(data[slice_idx, :, :, i]), np.abs(data_noisy[slice_idx, :, :, i])):.4f}", f"BM3D Denoised {rmse(np.abs(data[slice_idx, :, :, i]), bm4d_denoised[slice_idx, :, :, i]):.4f}", "Difference"]
    )


#%%

image_stack = np.abs(data_noisy)
min_val = np.min(image_stack)
max_val = np.max(image_stack)
image_stack = (image_stack - min_val) / (max_val - min_val)
p2s_denoised = patch2self(image_stack, alpha=1e-10, window_size=3)
p2s_denoised = p2s_denoised * (max_val - min_val) + min_val
print(f"Patch2Self Denoised shape: {p2s_denoised.shape}")
# p2s_denoised = np.expand_dims(p2s_denoised, axis=0)  # Add batch dimension
# data_noisy = np.expand_dims(data_noisy, axis=0)  # Add batch dimension
# data = np.expand_dims(data, axis=0)  # Add batch dimension

# %%


# slice_idx = 0  # Change this to visualize different slices

# for i in range(data.shape[-1]):
#     plot_images(
#         [np.abs(data[:, :, i]), np.abs(data_noisy[:, :, i]), np.abs(p2s_denoised[:, :, i]), np.abs(data_noisy[:, :, i] - p2s_denoised[:, :, i])],
#         ["Original", f"Noisy {rmse(data[:, :, i], data_noisy[:, :, i]):.4f}", f"Patch2Self Denoised {rmse(data[:, :, i], p2s_denoised[:, :, i]):.4f}", "Difference"]
#     )

slice_idx = 0  # Change this to visualize different slices

for i in range(data.shape[-1]):
    plot_images(
        [np.abs(data[slice_idx, :, :, i]), np.abs(data_noisy[slice_idx, :, :, i]), np.abs(p2s_denoised[slice_idx, :, :, i]), np.abs(data_noisy[slice_idx, :, :, i] - p2s_denoised[slice_idx, :, :, i])],
        ["Original", f"Noisy {rmse(data[slice_idx, :, :, i], data_noisy[slice_idx, :, :, i]):.4f}", f"Patch2Self Denoised {rmse(data[slice_idx, :, :, i], p2s_denoised[slice_idx, :, :, i]):.4f}", "Difference"]
    )


# %%
image_stack = np.abs(data_noisy)
min_val = np.min(image_stack)
max_val = np.max(image_stack)
image_stack = (image_stack - min_val) / (max_val - min_val)
# tucker_denoised = locally_low_rank_tucker(data_noisy, 0.5, tau=1.5, patch_transform="wavelet", lambda2d=1e-3)
tucker_denoised = locally_low_rank_tucker(image_stack, 0.6, tau=0.1, patch_transform="wavelet", lambda2d=1e-2, window_size=3, search_window=5)

tucker_denoised = tucker_denoised * (max_val - min_val) + min_val

# %%

# slice_idx = 6  # Change this to visualize different slices

# for i in range(data.shape[-1]):
#     plot_images(
#         [np.abs(data[:, :, i]), np.abs(data_noisy[:, :, i]), np.abs(tucker_denoised[:, :, i]), np.abs(data_noisy[:, :, i] - tucker_denoised[:, :, i])],
#         ["Original", f"Noisy {rmse(data[:, :, i], data_noisy[:, :, i]):.4f}", f"Tucker Denoised {rmse(data[:, :, i], tucker_denoised[:, :, i]):.4f}", "Difference"]
#     )


slice_idx = 0  # Change this to visualize different slices

for i in range(data.shape[-1]):
    plot_images(
        [np.abs(data[slice_idx, :, :, i]), np.abs(data_noisy[slice_idx, :, :, i]), np.abs(tucker_denoised[slice_idx, :, :, i]), np.abs(data_noisy[slice_idx, :, :, i] - tucker_denoised[slice_idx, :, :, i])],
        ["Original", f"Noisy {rmse(data[slice_idx, :, :, i], data_noisy[slice_idx, :, :, i]):.4f}", f"Tucker Denoised {rmse(data[slice_idx, :, :, i], tucker_denoised[slice_idx, :, :, i]):.4f}", "Difference"]
    )

# %%


from numba import njit


@njit
def linear_regression(A, b):
    """
    Perform linear regression using the normal equation method.

    Args:
        A: Design matrix (2D array).
        b: Target vector (1D array).

    Returns:
        Coefficients of the linear regression model.
    """
    w = np.linalg.lstsq(A, b)[0]
    return w

# %%


A = np.random.rand(100, 3) + 1j * np.random.rand(100, 3)
b = np.random.rand(100) + 1j * np.random.rand(100)
w = linear_regression(A.astype(np.complex64), b.astype(np.complex64))
print("Coefficients:", w)
# %%
