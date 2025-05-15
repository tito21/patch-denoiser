# %%
import timeit

import matplotlib.pyplot as plt
import numpy as np

from denoise.nlm_tensor import nlm_tensor, tensor_to_log_vector, log_vector_to_tensor, expm, logm, logm_explicit, expm_explicit


def plot_tensor(tensor, max, min):
    """Plot the tensor."""

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            axes[i, j].imshow(tensor[:, :, i, j], vmax=max, vmin=min)
            axes[i, j].axis("off")
    plt.show()



# tensor = np.load('tensor.npz')["tensor"]
tensor = np.load('/Users/adibiase/Documents/data/INVIVO/in_vivo_example_with_phase/Python_post_processing/debug/tensor.npy')

eigenvalues, eigenvectors = np.linalg.eigh(tensor)

mask = eigenvalues <= 0
mask = ~np.any(mask, axis=-1)

vmax = np.percentile(tensor, 90)
vmin = np.percentile(tensor, 10)

# %%

tensor_rec = expm(logm(tensor))
tensor_rec_explicit = expm_explicit(logm_explicit(tensor))
print("Tensor shape", tensor.shape)
print("Tensor rec shape", tensor_rec.shape)
print("Tensor rec explicit shape", tensor_rec_explicit.shape)
print(np.allclose(tensor, tensor_rec))
print(np.allclose(tensor, tensor_rec_explicit))
print(np.allclose(tensor_rec, tensor_rec_explicit))



tensor_log = tensor_to_log_vector(tensor)
tensor_rec = log_vector_to_tensor(tensor_log)
print("Tensor shape", tensor.shape)
print("Tensor log shape", tensor_log.shape)
print("Tensor rec shape", tensor_rec.shape)
print(np.allclose(tensor, tensor_rec))

plot_tensor(tensor_rec[0], vmax, vmin)

# %%

tensor_denoised = nlm_tensor(np.ascontiguousarray(tensor).copy(), mask=mask, patch_size=5, window_size=15, h=1)

# %%

plot_tensor(tensor[0], vmax, vmin)
plot_tensor(tensor_denoised[0], vmax, vmin)
# %%
