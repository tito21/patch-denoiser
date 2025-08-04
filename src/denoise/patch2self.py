import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import Ridge



def search_patches(image_padded: NDArray, local_patch: NDArray, tau: float, k_start: int, k_end: int, i_start: int, i_end: int, j_start: int, j_end: int, search_window: int, window_size: int) -> NDArray:
    """
    Extract patches from the padded image data.

    Parameters:
        image_padded (NDArray): Padded image data.
        local_patch (NDArray): Local patch to compare against.
        tau (float): Threshold for patch similarity.
        k_start (int): Start index for the z-axis.
        k_end (int): End index for the z-axis.
        i_start (int): Start index for the x-axis.
        i_end (int): End index for the x-axis.
        j_start (int): Start index for the y-axis.
        j_end (int): End index for the y-axis.
        search_window (int): Size of the search window.
        window_size (int): Size of the window for patch extraction.

    Returns:
        NDArray: Extracted patches.
    """
    # patches = []
    indices = []
    for k in range(k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                patch = image_padded[k:k + window_size, i:i + window_size, j:j + window_size, :]
                if np.sum(np.abs(patch - local_patch)**2) < tau:
                    # patches.append(patch.reshape(window_size**3, -1))
                    indices.append((k, i, j))
    return indices # Shape: (num_patches * channels, patch_size^3)


def patch2self_non_local(
    data: ArrayLike,
    alpha: float = 1e-6,
    tau: float = 0.1,
    window_size: int = 3,
    search_window: int = 5,
) -> NDArray:
    """
    Denoise an image using the Patch2Self algorithm.

    Parameters:
        data (ArrayLike): Input image data.
        alpha (float): Regularization parameter.
        window_size (int): Size of the window for patch extraction.

    Returns:
        NDArray: Denoised image data.
    """

    data = np.array(data, dtype=np.float32)
    n_slices, n_row, n_col, n_channels = data.shape

    data_padded = np.pad(
        data,
        (
            (window_size // 2, window_size // 2),
            (window_size // 2, window_size // 2),
            (window_size // 2, window_size // 2),
            (0, 0),
        ),
        mode="reflect",
    )
    n_volumes = data_padded.shape[0] * data_padded.shape[1] * data_padded.shape[2]
    patches = np.lib.stride_tricks.sliding_window_view(data_padded, (window_size, window_size, window_size, data.shape[-1]))
    patches = patches.reshape(data_padded.shape[0], data_padded.shape[1], data_padded.shape[2], window_size**3, data.shape[-1])
    print(patches.shape)
    print(f"Extracted patches shape: {patches.shape}")

    numerator = np.zeros_like(data_padded, dtype=np.float32)
    denominator = np.zeros_like(data_padded, dtype=np.float32)

    for slice_index in range(data.shape[0]):
        for row in range(data.shape[1]):
            for col in range(data.shape[2]):
                k_start = slice_index - min(search_window, slice_index)
                k_end = slice_index + min(search_window + 1, n_slices - slice_index)
                i_start = row - min(search_window, row)
                i_end = row + min(search_window + 1, n_row - row)
                j_start = col - min(search_window, col)
                j_end = col + min(search_window + 1, n_col - col)
                local_patch = data_padded[slice_index:slice_index + window_size, row:row + window_size, col:col + window_size, :]
                indices = search_patches(data_padded, local_patch, tau, k_start, k_end, i_start, i_end, j_start, j_end, search_window, window_size)

                weights = 1 / len(indices) if len(indices) > 0 else 1
                local_volume = []
                for idx, (k, i, j) in enumerate(indices):
                    local_volume.append(patches[k, i, j, :, :])
                print(local_volume.shape)
                local_volume = np.array(local_volume, dtype=np.float32).transpose(0, 2, 1).reshape(-1, local_volume.shape[2])
                for c in range(local_volume.shape[0]):
                    curr_x = local_volume[np.arange(local_volume.shape[0]) != c].reshape(-1, local_volume.shape[1])
                    y = local_volume[c, local_volume.shape[1] // 2, :]
                    model = Ridge(alpha=alpha)
                    model.fit(curr_x.T, y.T)

                    print(model.predict(curr_x.T).reshape(data.shape[:-1]))

                    for idx, (k, i, j) in enumerate(indices):
                        numerator[slice_index:slice_index + window_size, row:row + window_size, col:col + window_size, :] += weights * model.predict(curr_x.T).reshape(data.shape[:-1])
                        denominator[slice_index:slice_index + window_size, row:row + window_size, col:col + window_size, c] += weights

    denoised = denominator / (numerator + 1e-8)  # Avoid division by zero
    return denoised[
        window_size // 2 + 1 : -window_size // 2,
        window_size // 2 + 1 : -window_size // 2,
        window_size // 2 + 1 : -window_size // 2,
    ]

def patch2self(
    data: ArrayLike,
    alpha: float = 1e-10,
    window_size: int = 3,
) -> NDArray:
    """
    Denoise an image using the Patch2Self algorithm.

    Parameters:
        data (ArrayLike): Input image data.
        alpha (float): Regularization parameter.
        window_size (int): Size of the window for patch extraction.

    Returns:
        NDArray: Denoised image data.
    """

    original_shape = data.shape
    data = np.array(data, dtype=np.float32)
    if data.ndim == 3:
        data = data[np.newaxis, :, :, :]  # Add channel dimension if missing

    data_padded = np.pad(
        data,
        (
            (window_size // 2, window_size // 2),
            (window_size // 2, window_size // 2),
            (window_size // 2, window_size // 2),
            (0, 0),
        ),
        mode="reflect",
    )
    denoised = np.zeros_like(data, dtype=np.float32)
    n_volumes = data_padded.shape[0] * data_padded.shape[1] * data_padded.shape[2]
    patches = np.lib.stride_tricks.sliding_window_view(data_padded, (window_size, window_size, window_size, data.shape[-1]))
    patches = patches.reshape(-1, window_size**3, data.shape[-1])
    patches = patches.transpose(2, 1, 0)

    for c in range(data.shape[-1]):
        curr_x = patches[np.arange(patches.shape[0]) != c].reshape(-1, patches.shape[2])
        y = patches[c, patches.shape[1] // 2, :]
        model = Ridge(alpha=alpha)
        model.fit(curr_x.T, y.T)
        denoised[:, :, :, c] = model.predict(curr_x.T).reshape(data.shape[:-1])

    return denoised.reshape(original_shape)