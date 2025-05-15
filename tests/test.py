# %%

import timeit


import matplotlib.pyplot as plt
import numpy as np

from phantom import phantom_multicontrast
from denoise.patch_denoise import locally_low_rank_tucker, tucker, tucker_to_tensor

def plot_images(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 15))
    for ax, img, title in zip(axes, images, titles):
        # ax.imshow(img, cmap='gray')
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.show()


def randn_complex(shape):
    """Generate a random complex tensor of given shape."""
    return np.random.randn(*shape) + 1j * np.random.randn(*shape)


# %%

shape = 25, 25, 25
r_shape = 5, 5, 5
C = randn_complex(r_shape)
# Generate orthogonal factor matrices
A1, _ = np.linalg.qr(randn_complex((shape[0], r_shape[0])))
A2, _ = np.linalg.qr(randn_complex((shape[1], r_shape[1])))
A3, _ = np.linalg.qr(randn_complex((shape[2], r_shape[2])))

# Compute Tucker tensor
tensor = tucker_to_tensor(C, [A1, A2, A3])


print("Tucker tensor shape", tensor.shape)
# tensor += 0.1 * randn_complex(shape)

core, factors = tucker(tensor, full_svd=True)
print("Full svd error", np.linalg.norm(tucker_to_tensor(core, factors).ravel() - tensor.ravel()) / np.linalg.norm(tensor.ravel()))
core, factors = tucker(tensor, full_svd=False, rank=np.asarray([tensor.shape[0] // 2, tensor.shape[1] // 2, tensor.shape[2] // 2]))
print("Truncated svd error", np.linalg.norm(tucker_to_tensor(core, factors).ravel() - tensor.ravel()) / np.linalg.norm(tensor.ravel()))

print("Core shape", core.shape)
# print(core)
print("Factor shapes", [f.shape for f in factors])

# fig, axs = plt.subplots(1, len(core), figsize=(15, 5))
# for i, ax in enumerate(axs):
#     ax.imshow(np.abs(core[i, :, :]), cmap='gray')
#     ax.set_title(f"Core slice {i}")
#     ax.axis('off')
# plt.show()

# print("Old", timeit.timeit(
#     "tucker_to_tensor(core, factors)",
#     globals=globals(),
#     number=100,
# ))


# %%


data = phantom_multicontrast(32)
noise_level = 0.1
data_noisy = data + noise_level * (np.random.randn(*data.shape) + 1j * np.random.randn(*data.shape))


# %%


print("identity", timeit.timeit(
    'locally_low_rank_tucker(data_noisy, 3, tau=0.5/noise_level, patch_transform="identity", lambda2d=0.1)',
    globals=globals(),
    number=5,
))

# %%

print("identity", timeit.timeit(
    'locally_low_rank_tucker(data_noisy, 3, tau=0.5/noise_level, patch_transform="identity", lambda2d=0.1)',
    globals=globals(),
    number=5,
))

# %%


print("fft", timeit.timeit(
    'locally_low_rank_tucker(data_noisy, 3, tau=0.5/noise_level, patch_transform="fft", lambda2d=0.1)',
    globals=globals(),
    number=5,
))

# %%

print("wavelet", timeit.timeit(
    'locally_low_rank_tucker(data_noisy, 3, tau=0.5/noise_level, patch_transform="wavelet", lambda2d=0.1)',
    globals=globals(),
    number=5,
))