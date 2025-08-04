from numba import njit, objmode, float32, complex64, prange
from numba_progress import ProgressBar
import numpy as np
from numpy.typing import ArrayLike, NDArray

# from numpy.fft import fft, ifft, fftn, ifftn
from scipy.fft import fft, ifft, fftn, ifftn, fftshift
from scipy.fft import dct, idct
from pywt import wavedecn, waverecn, array_to_coeffs, coeffs_to_array


@njit
def dct2(x):
    return dct(dct(x, axis=0), axis=1)


@njit
def idct2(x):
    return idct(idct(x, axis=0), axis=1)


@njit
def dct3(x):
    return dct(dct(dct(x, axis=0), axis=1), axis=2)


@njit
def idct3(x):
    return idct(idct(idct(x, axis=0), axis=1), axis=2)


@njit
def fft2(x):
    return fft(fft(x, axis=0), axis=1)


@njit
def ifft2(x):
    return ifft(ifft(x, axis=0), axis=1)


@njit(complex64[:, :, :](complex64[:, :, :]))
def fft3(
    x: np.ndarray[tuple[int, int, int], complex | float],
) -> np.ndarray[tuple[int, int, int], complex]:
    return fftn(x, axes=(0, 1, 2))


@njit(complex64[:, :, :](complex64[:, :, :]))
def ifft3(
    x: np.ndarray[tuple[int, int, int], complex],
) -> np.ndarray[tuple[int, int, int], complex]:
    return ifftn(x, axes=(0, 1, 2))


@njit(complex64[:, :, :, :](complex64[:, :, :, :]))
def fft4(
    x: np.ndarray[tuple[int, int, int, int], complex | float],
) -> np.ndarray[tuple[int, int, int, int], complex]:
    return fftn(x, axes=(0, 1, 2, 3))


@njit(complex64[:, :, :, :](complex64[:, :, :, :]))
def ifft4(
    x: np.ndarray[tuple[int, int, int, int], complex],
) -> np.ndarray[tuple[int, int, int, int], complex]:
    return ifftn(x, axes=(0, 1, 2, 3))


@njit
def wavelet_denoise(x: NDArray, wavelet="bior1.3", threshold=0.1, axis=(0, 1)):
    """
    Perform wavelet denoising on the input array using the specified wavelet and threshold.

    Args:
        x: Input array to be denoised.
        wavelet: The type of wavelet to use for the transform. Default is 'db3'.
        threshold: The threshold value for denoising. Default is 0.1.

    Returns:
        x_denoised: The denoised array after applying the wavelet transform and thresholding.
    """
    with objmode(rec="complex64[:, :, :, :]"):
        coeffs = wavedecn(x, wavelet, axes=axis)
        coeffs_arr, coeffs_slices = coeffs_to_array(coeffs, axes=axis)
        # print("coeffs_arr max", np.abs(coeffs_arr).max(), "coeffs_arr min", np.abs(coeffs_arr).min(), "coeffs_arr mean", np.abs(coeffs_arr).mean(), "coeffs_arr std", np.abs(coeffs_arr).std())
        coeffs_arr = hard_thresholding(coeffs_arr, threshold)
        rec = waverecn(array_to_coeffs(coeffs_arr, coeffs_slices), wavelet, axes=axis)
    return rec


@njit(fastmath=True)
def hard_thresholding(signal: ArrayLike, threshold: float) -> NDArray:
    """
    Apply hard thresholding to a signal.

    Hard thresholding sets elements of the signal to zero if their absolute
    value is less than the specified threshold. Elements with an absolute
    value greater than or equal to the threshold remain unchanged.

    Args:
        signal: The input signal to be thresholded.
        threshold: The threshold value. Elements with an absolute
                           value less than this will be set to zero.

    Returns:
        out: The thresholded signal, where elements below the
                       threshold are set to zero.
    """
    # return np.where(signal < threshold, 0, signal)
    return np.where(np.abs(signal) < threshold, 0, signal).astype(signal.dtype)


@njit(complex64[:, :, :](complex64[:, :, :], float32), fastmath=True)
def low_pass_filter3d(signal: NDArray, sigma: float) -> NDArray:
    """
    Apply a low-pass filter to the input array.

    Args:
        X: Input array to be filtered.
        sigma: Standard deviation of the Gaussian filter.

    Returns:
        out: The filtered array.
    """
    X = np.arange(signal.shape[0])[:, np.newaxis, np.newaxis]
    Y = np.arange(signal.shape[1])[np.newaxis, :, np.newaxis]
    Z = np.arange(signal.shape[2])[np.newaxis, np.newaxis, :]
    center = (signal.shape[0] // 2, signal.shape[1] // 2, signal.shape[2] // 2)
    X -= center[0]
    Y -= center[1]
    Z -= center[2]
    f = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2)).astype(signal.dtype)
    with objmode(f_shifted="complex64[:, :, :]"):  # annotate return type
        f_shifted = fftshift(f)
    return f_shifted * signal


@njit(complex64[:, :, :, :](complex64[:, :, :, :], float32), fastmath=True)
def low_pass_filter4d(signal: NDArray, sigma: float) -> NDArray:
    """
    Apply a low-pass filter to the input array.

    Args:
        X: Input array to be filtered.
        sigma: Standard deviation of the Gaussian filter.

    Returns:
        out: The filtered array.
    """
    X = np.arange(signal.shape[0])[:, np.newaxis, np.newaxis, np.newaxis]
    Y = np.arange(signal.shape[1])[np.newaxis, :, np.newaxis, np.newaxis]
    Z = np.arange(signal.shape[2])[np.newaxis, np.newaxis, :, np.newaxis]
    W = np.arange(signal.shape[3])[np.newaxis, np.newaxis, np.newaxis, :]
    center = (signal.shape[0] // 2, signal.shape[1] // 2, signal.shape[2] // 2, signal.shape[3] // 2)
    X -= center[0]
    Y -= center[1]
    Z -= center[2]
    W -= center[3]
    f = np.exp(-(X**2 + Y**2 + Z**2 + W**2) / (2 * sigma**2)).astype(signal.dtype)
    with objmode(f_shifted="complex64[:, :, :, :]"):  # annotate return type
        f_shifted = fftshift(f)
    return f_shifted * signal


@njit(fastmath=True)
def truncated_svd(A, rank, over_sample=10, power_iteration=1):
    """Compute the truncated SVD of a matrix A using randomized SVD.
    The algorithm is based on the scikit-learn implementation.

    """
    n_samples, n_features = A.shape
    transpose = False
    if n_samples < n_features:
        transpose = True
    if transpose:
        A = np.ascontiguousarray(np.conj(A).T)
    P = (
        np.random.randn(A.shape[1], rank + over_sample)
        + 1j * np.random.randn(A.shape[1], rank + over_sample)
    ).astype(A.dtype)
    Z = A @ P
    At = np.ascontiguousarray(np.conj(A.T))
    for k in range(power_iteration):
        Z = A @ (At @ Z)

    Q, R = np.linalg.qr(Z)

    Y = np.ascontiguousarray(np.conj(Q.T)) @ A
    UY, S, Vh = np.linalg.svd(Y, full_matrices=False)
    U = Q @ UY

    if transpose:
        # transpose back the results according to the input convention
        return np.conj(Vh[:rank, :]).T, S[:rank], np.conj(U[:, :rank]).T
    else:
        return U[:, :rank], S[:rank], Vh[:rank, :]


@njit
def fold(tensor: ArrayLike, mode: int, shape: ArrayLike) -> NDArray:
    """
    Rearranges a flattened tensor into a multidimensional array based on the specified mode and shape.

    Args:
        tensor: The input flattened tensor to be reshaped and rearranged.
        mode: The mode of folding, which determines the axis along which the tensor is unfolded.
              Must be one of {0, 1, 2}.
        shape: The original shape of the multidimensional array before flattening.

    Returns:
        out: A contiguous array reshaped and rearranged according to the specified mode and shape.

    Raises:
        ValueError: If the mode is not one of {0, 1, 2}.
    """
    full_shape = np.asarray(shape)
    mode_dim = full_shape[mode]
    full_shape = np.delete(full_shape, mode)
    full_shape = [mode_dim] + [full_shape[i] for i in range(len(full_shape))]
    full_shape = (full_shape[0], full_shape[1], full_shape[2])
    if mode == 0:
        return np.ascontiguousarray(np.reshape(tensor, full_shape))
    elif mode == 1:
        return np.ascontiguousarray(
            np.transpose(np.reshape(tensor, full_shape), (1, 0, 2))
        )
    elif mode == 2:
        return np.ascontiguousarray(
            np.transpose(np.reshape(tensor, full_shape), (1, 2, 0))
        )
    else:
        raise ValueError("mode should be in {0, 1, 2}")


@njit
def unfold(tensor: ArrayLike, mode: int) -> NDArray:
    """
    Unfolds a 3D tensor along the specified mode into a 2D matrix.

    Args:
        tensor: A 3D numpy array to be unfolded.
        mode: The mode along which to unfold the tensor.
                    Must be one of {0, 1, 2}.
                    - mode=0: Unfold along the first dimension.
                    - mode=1: Unfold along the second dimension.
                    - mode=2: Unfold along the third dimension.

    Returns:
        numpy.ndarray: A 2D numpy array resulting from unfolding the input tensor.

    Raises:
        ValueError: If the mode is not one of {0, 1, 2}.
    """
    if mode == 0:
        return np.ascontiguousarray(np.reshape(tensor, (tensor.shape[0], -1))).astype(tensor.dtype)
    elif mode == 1:
        return np.ascontiguousarray(
            np.reshape(
                np.ascontiguousarray(np.transpose(tensor, (1, 0, 2))),
                (tensor.shape[1], -1),
            )
        ).astype(tensor.dtype)
    elif mode == 2:
        return np.ascontiguousarray(
            np.reshape(
                np.ascontiguousarray(np.transpose(tensor, (2, 0, 1))),
                (tensor.shape[2], -1),
            )
        ).astype(tensor.dtype)
    else:
        raise ValueError("mode should be in {0, 1, 2}")


@njit
def multi_mode_dot(tensor, factors, skip=-1):
    for i in range(3):
        if skip == i:
            continue
        tensor = mode_dot(tensor, factors[i], i)
    return tensor


@njit
def multi_mode_dot_transpose(tensor, factors, skip=-1):
    for i in range(3):
        if skip == i:
            continue
        tensor = mode_dot(tensor, np.conj(factors[i].T), i)
    return tensor


@njit(fastmath=True)
def mode_dot(tensor, m, mode):
    new_shape = list(tensor.shape)
    new_shape[mode] = m.shape[0]
    res = np.dot(m, unfold(tensor, mode))
    return fold(res, mode, new_shape)


@njit
def tucker_to_tensor(core, factors):
    tensor = core
    for i in range(3):
        tensor = mode_dot(tensor, factors[i], i)
    return tensor


@njit(fastmath=True)
def tucker(
    tensor: ArrayLike, full_svd=True, rank=np.array([25, 25, 25])
) -> tuple[NDArray, list[NDArray]]:
    """
    Tucker decomposition of a 3D tensor.
    Args:
        tensor: A 3D numpy array to be decomposed.
        full_svd: If True, use full SVD. If False, use truncated SVD.
        rank: The rank of the decomposition for each mode.
              If full_svd is True, this parameter is ignored. The current default is arbitrary.

    Returns:
        core: core tensor
        factor: list factor matrices.
    """
    # if tensor.size > 100:
    #     full_svd = False
    if full_svd:
        factors = [
            np.empty((tensor.shape[i], tensor.shape[i]), dtype=tensor.dtype)
            for i in range(3)
        ]
    else:
        factors = [
            np.empty((tensor.shape[i], rank[i]), dtype=tensor.dtype) for i in range(3)
        ]
    # factors = []

    for i in range(3):
        # U, _, _ = truncated_svd(unfold(tensor, i), rank=rank[i])
        if full_svd:
            U, _, _ = np.linalg.svd(unfold(tensor, i), full_matrices=False)
            factors[i] = np.ascontiguousarray(U).astype(tensor.dtype)
        else:
            U, _, _ = truncated_svd(unfold(tensor, i), rank=rank[i])
            factors[i] = np.ascontiguousarray(U).astype(tensor.dtype)
        # print("U", U.shape, U.dtype)
        # factors.append(U)

    core = multi_mode_dot_transpose(tensor, factors)
    return core, factors


@njit(fastmath=True)
def compress_wavelet(tensor: ArrayLike, threshold: float) -> NDArray:
    """
    Compresses a given tensor using wavelet decomposition and hard thresholding.

    This function performs wavelet decomposition on the input tensor, applies
    hard thresholding to the coefficients to retain only significant values,
    and reconstructs the tensor from the compressed coefficients.

    Args:
        tensor: The input tensor to be compressed.
        threshold: The threshold value for hard thresholding. Elements in the coefficients with absolute values below
        this threshold are set to zero.

    Returns:
        compressed_tensor: The reconstructed tensor after compression.
    """
    with objmode(coeffs="complex64[:, :, :]"):
        coeffs = wavedecn(tensor, "bior1.3")
        coeffs_arr, coeffs_slices = coeffs_to_array(coeffs)
        coeffs_arr = hard_thresholding(coeffs_arr, threshold)
        rec = waverecn(array_to_coeffs(coeffs_arr, coeffs_slices), "bior1.3")
        N = np.sum(coeffs_arr != 0)
    return rec, N


@njit
def compress_tucker(tensor: ArrayLike, threshold: float) -> tuple[NDArray, int]:
    """
    Compresses a given tensor using Tucker decomposition and hard thresholding.

    This function performs Tucker decomposition on the input tensor, applies
    hard thresholding to the core tensor to retain only significant values,
    and reconstructs the tensor from the compressed core and factors.

    Args:
        tensor: The input tensor to be compressed.
        threshold: The threshold value for hard thresholding. Elements in the core tensor with absolute values below
        this threshold are set to zero.

    Returns:
        compressed_tensor: The reconstructed tensor after compression.
        N: The number of non-zero elements in the compressed core tensor.
    """
    rank = np.array([tensor.shape[0] // 2 + 1, tensor.shape[1] // 2 + 1, tensor.shape[2] // 2 + 1], dtype=np.uint16)
    core, factors = tucker(tensor, rank=rank, full_svd=False)
    core = hard_thresholding(core, threshold)
    N = np.sum(core != 0)
    return tucker_to_tensor(core, factors), N


@njit(nogil=True, fastmath=True)
def weighted_dist(x, y, w):
    return np.sum(w * (x - y) ** 2)


@njit(nogil=True, fastmath=True)
def search_windows_llr_hd(
    local_patch: NDArray,
    padded_image: NDArray,
    window_size: int,
    k_start: int,
    k_end: int,
    i_start: int,
    i_end: int,
    j_start: int,
    j_end: int,
    patch_transform: str = "fft",
    lambda2d: float = 0.1,
    tau: float = 0.01,
) -> tuple[NDArray, NDArray]:
    """
    Searches for similar patches within a specified window in a padded image using
    local low-rank hard thresholding and a distance threshold.

    Args:
        local_patch: The reference patch to compare against,
            typically a 3D array representing a small region of the image.
        padded_image: The padded image from which patches are extracted.
        window_size: The size of the square window to extract patches.
        i_start: The starting index along the first dimension of the search window.
        i_end: The ending index (exclusive) along the first dimension of the search window.
        j_start: The starting index along the second dimension of the search window.
        j_end: The ending index (exclusive) along the second dimension of the search window.
        patch_transform: The type of transform to apply to the patches. Currently only "fft" and "identity" are supported.
        lambda2d: The threshold parameter for hard thresholding in the frequency domain.
        tau: The distance threshold for determining patch similarity.

    Returns:
        patches: A contiguous array of similar patches found within the search window.
        indicies: An array of indices corresponding to the positions of the similar patches
    """

    patches = np.empty(
        (
            (k_end - k_start) * (i_end - i_start) * (j_end - j_start),
            local_patch.shape[0],
            local_patch.shape[1],
            local_patch.shape[2],
            local_patch.shape[3],
        ),
        dtype=local_patch.dtype,
    )
    indices = np.empty(((k_end - k_start) * (i_end - i_start) * (j_end - j_start), 3), dtype=np.uint16)

    if patch_transform == "fft":
        # print("Using FFT")
        # local_patch = hard_thresholding(wavelet_transform(local_patch)[1], lambda2d)
        local_patch_transformed = (
            low_pass_filter4d(fft4(local_patch), lambda2d)
            / window_size**3
            # hard_thresholding(fft3(local_patch), lambda2d) / window_size**2
        )
        # print("local_patch", local_patch.shape, local_patch.dtype)
    elif patch_transform == "wavelet":
        local_patch_transformed = wavelet_denoise(local_patch, threshold=lambda2d, axis=(0, 1, 2))
    else:
        local_patch_transformed = local_patch

    ii = 0
    dist_list = np.zeros((k_end - k_start) * (i_end - i_start) * (j_end - j_start), dtype=np.float32)
    for k in range(k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                patch = padded_image[k : k + window_size, i : i + window_size, j : j + window_size, :]
                if patch_transform == "fft":
                    # patch = hard_thresholding(wavelet_transform(patch)[1], lambda2d)
                    patch_transformed = (
                        low_pass_filter4d(fft4(patch), lambda2d)
                        / window_size**3
                        # hard_thresholding(fft3(patch), lambda2d) / window_size**2
                    )
                elif patch_transform == "wavelet":
                    patch_transformed = wavelet_denoise(patch, threshold=lambda2d, axis=(0, 1, 2))
                else:
                    patch_transformed = patch
                # print("patch", patch_transformed.shape, patch_transformed.dtype, "local_patch", local_patch_transformed.shape, local_patch_transformed.dtype)
                dist = (
                    np.sum(np.abs(local_patch_transformed - patch_transformed) ** 2)
                ) / window_size**3

                dist_list[ii] = dist
                # if dist < 0.01:
                #     print("Distance", dist)

                if dist < tau:
                    patches[ii, :, :, :, :] = patch
                    indices[ii, 0] = k
                    indices[ii, 1] = i
                    indices[ii, 2] = j
                    ii += 1
    # print("Number of patches found", ii, "/", patches.shape[0], patches[:ii, :, :, :].shape)
    # print("Distance list mean, std", np.mean(dist_list[:ii]), np.std(dist_list[:ii]))
    return np.ascontiguousarray(patches[:ii, :, :, :, :]), indices[:ii, :]


@njit(parallel=True, nogil=True)
def _main_loop_llr_tucker(
    progress_bar: ProgressBar,
    noisy_image: NDArray,
    padded_image: NDArray,
    window_size: int,
    patch_transform: str,
    threshold: float,
    lambda2d: float,
    search_window: int,
    tau: float,
) -> NDArray:
    """
    Perform the main loop for low-rank Tucker decomposition-based denoising.

    This function processes a noisy image using a sliding window approach,
    applying higher-order singular value decomposition (HOSVD) to patches
    within a search window to perform denoising.

    Args:
        progress_bar: A progress bar object to track the loop's progress.
        noisy_image: The noisy input image of shape (height, width, channels).
        padded_image: The padded version of the noisy image to handle boundary conditions.
        window_size: The size of the local patch window.
        patch_transform: The type of transform to apply to the patches. Currently only
            "fft", "wavelet" and "identity" are supported.
        threshold: Threshold value for Tucker decomposition.
        lambda2d: Regularization parameter for 2D low-rank approximation.
        search_window (int): The size of the search window for finding similar patches.
        tau: Parameter controlling the similarity threshold for patch selection.

    Returns:
        denoised: The denoised image, cropped to remove padding.
    """

    n_slices, n_row, n_col, n_channels = (
        noisy_image.shape[0],
        noisy_image.shape[1],
        noisy_image.shape[2],
        noisy_image.shape[3],
    )

    numerator = np.zeros_like(padded_image)
    denominator = np.zeros_like(padded_image)
    total = n_slices * n_row * n_col
    for t in prange(total):

        slice_index = t // (n_row * n_col)
        row = (t // n_col) % n_row
        col = t % n_col
        slice_index = int(slice_index)
        row = int(row)
        col = int(col)

        k_start = slice_index - min(search_window, slice_index)
        k_end = slice_index + min(search_window + 1, n_slices - slice_index)
        i_start = row - min(search_window, row)
        i_end = row + min(search_window + 1, n_row - row)
        j_start = col - min(search_window, col)
        j_end = col + min(search_window + 1, n_col - col)
        local_patch = padded_image[
            slice_index : slice_index + window_size,
            row : row + window_size,
            col : col + window_size,
            :,
        ]
        patches, indicies = search_windows_llr_hd(
            local_patch,
            padded_image,
            window_size,
            k_start,
            k_end,
            i_start,
            i_end,
            j_start,
            j_end,
            patch_transform,
            lambda2d,
            tau,
        )
        # print("patches", patches.shape, patches.dtype)
        # Higher order SVD
        patches = np.ascontiguousarray(np.transpose(patches, (1, 2, 3, 0, 4)))
        patches = patches.reshape(window_size * window_size * window_size, indicies.shape[0], n_channels)
        patches_rec, N = compress_tucker(
            patches,
            threshold,
        )
        patches_rec = np.ascontiguousarray(patches_rec.reshape(window_size, window_size, window_size, indicies.shape[0], n_channels))
        patches_rec = np.ascontiguousarray(np.transpose(patches_rec, (3, 0, 1, 2, 4)))
        weights = 1 / (1 + N)
        patches = patches_rec * weights
        for (kk, ii, jj), patch in zip(indicies, patches):
            numerator[kk : kk + window_size, ii : ii + window_size, jj : jj + window_size] += patch
            denominator[kk : kk + window_size, ii : ii + window_size, jj : jj + window_size] += weights

        progress_bar.update(1)

    denoised = numerator / (denominator + 1e-10)
    return denoised[
        window_size // 2 + 1 : -window_size // 2,
        window_size // 2 + 1 : -window_size // 2,
        window_size // 2 + 1 : -window_size // 2,
        :,
    ]


def locally_low_rank_tucker(
    noisy_image: ArrayLike,
    threshold: float,
    tau: float = 0.1,
    patch_transform: str = "fft",
    lambda2d: float = 1,
    window_size: int = 5,
    search_window: int = 11,
) -> NDArray:
    """
    Perform denoising on a noisy image using the Locally Low-Rank Tucker decomposition method.

    Args:
        noisy_image: The input noisy image as a 3D NumPy array (height, width, channels).
        threshold: Threshold value for Tucker decomposition.
        tau: Parameter controlling the similarity threshold for patch selection. A good default for the "identity" mode
             is the inverse of the noise level (1/sigma). For the "fft" mode, a value of 0.5/sigma is recommended.
        patch_transform: The type of transform to apply to the patches. Currently only "fft", "wavelet" and "identity" are supported.
        lambda2d: Weighting parameter for 2D low-rank approximation. Default is 1.
        window_size: Size of the local patch window. Default is 5.
        search_window: Size of the search window for similar patches. Default is 11.

    Returns:
        denoised_image: The denoised image as a 3D NumPy array with the same shape as the input.

    Notes:
        - The function pads the input image to handle boundary conditions using reflection padding.
        - The denoising process is performed iteratively over the image using a progress bar to track progress.
        - The `_main_loop_llr_tucker` function is used internally to perform the core denoising operation.

    """

    if window_size % 2 == 0:
        window_size += 1  # Ensure window size is odd
    if search_window % 2 == 0:
        search_window += 1  # Ensure search window is odd

    original_shape = noisy_image.shape
    noisy_image = np.asarray(noisy_image)
    if noisy_image.ndim == 3:
        noisy_image = np.expand_dims(noisy_image, axis=0)
    padded_image = np.pad(
        noisy_image,
        (
            (window_size // 2 + 1, window_size // 2 + 1),
            (window_size // 2 + 1, window_size // 2 + 1),
            (window_size // 2 + 1, window_size // 2 + 1),
            (0, 0),
        ),
        "reflect",
    ).astype(np.complex64)

    n_slices, n_row, n_col = noisy_image.shape[0], noisy_image.shape[1], noisy_image.shape[2]
    with ProgressBar(total=n_slices * n_row * n_col) as progress_bar:
        denoised_image = _main_loop_llr_tucker(
            progress_bar,
            noisy_image.astype(np.complex64),
            padded_image,
            window_size,
            patch_transform,
            threshold,
            lambda2d,
            search_window,
            tau,
        )
    if not np.iscomplexobj(noisy_image):
        denoised_image = denoised_image.real

    return denoised_image.reshape(original_shape)

@njit
def select_patches(row, col, signal, patch, search_window=25, tau=0.005):
    patch_size = patch.shape[0]
    n_row, n_col, n_channels = signal.shape
    n_row = n_row - patch_size - 1
    n_col = n_col - patch_size - 1
    i_start = row - min(search_window, row)
    i_end = row + min(search_window + 1, n_row - row)
    j_start = col - min(search_window, col)
    j_end = col + min(search_window + 1, n_col - col)
    patches = np.empty(
        ((i_end - i_start) * (j_end - j_start), patch_size, patch_size, n_channels),
        dtype=signal.dtype,
    )
    indicies = np.empty(((i_end - i_start) * (j_end - j_start), 2), dtype=np.int32)
    ii = 0
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            current_patch = signal[i : i + patch_size, j : j + patch_size]
            dist = np.sum(np.abs(current_patch - patch) ** 2) / patch_size**2
            if dist < tau:
                patches[ii] = current_patch
                indicies[ii, 0] = i
                indicies[ii, 1] = j
                ii += 1
    return patches[:ii], indicies[:ii]


@njit
def select_patches_indicies(signal, indicies, patch_size):
    patches = np.empty(
        (len(indicies), patch_size, patch_size, signal.shape[-1]), dtype=signal.dtype
    )
    for ii in range(len(indicies)):
        i, j = indicies[ii]
        patches[ii] = signal[i : i + patch_size, j : j + patch_size, :]
    return patches


@njit
def collaborative_filtering_wiener(patches, original, sigma=0.1):
    core, _ = tucker(patches)
    W = np.abs(core) ** 2 / (np.abs(core) ** 2 + sigma**2)
    # patches_rec = ifft3(W * fft3(original)).real
    # patches_rec = ifft3(W * patches_dct).real
    core, factors = tucker(original)
    patches_rec = tucker_to_tensor(W * core, factors)
    return np.ascontiguousarray(patches_rec), np.ascontiguousarray(W)


@njit
def _wiener_main_loop(
    progress,
    noisy_image,
    padded_noisy,
    padded_estimate,
    sigma,
    tau,
    patch_size,
    search_window,
):
    numerator = np.zeros_like(padded_noisy)
    denominator = np.zeros(padded_noisy.shape, dtype=np.float32)

    n_row, n_col, n_channels = (
        noisy_image.shape[0],
        noisy_image.shape[1],
        noisy_image.shape[2],
    )

    for i in range(0, n_row):
        for j in range(0, n_col):
            patch = padded_estimate[i : i + patch_size, j : j + patch_size]
            patches_estimate, indicies = select_patches(
                i, j, padded_estimate, patch, search_window, tau
            )
            patches_noisy = select_patches_indicies(padded_noisy, indicies, patch_size)
            # patches_noisy = np.zeros_like(patches_estimate)

            patches_rec, W = collaborative_filtering_wiener(
                np.ascontiguousarray(
                    patches_estimate.reshape(-1, patch_size * patch_size, n_channels)
                ),
                np.ascontiguousarray(
                    patches_noisy.reshape(-1, patch_size * patch_size, n_channels)
                ),
                sigma,
            )

            weights = 1 / (sigma**2 * np.sum(W**2))
            patches_rec = patches_rec * weights
            for (ii, jj), patch in zip(indicies, patches_rec):
                numerator[ii : ii + patch_size, jj : jj + patch_size] += patch
                denominator[ii : ii + patch_size, jj : jj + patch_size] += weights
            progress.update(1)

    denoised = numerator / (denominator + 1e-10)
    return denoised[
        patch_size // 2 + 1 : -patch_size // 2, patch_size // 2 + 1 : -patch_size // 2
    ]


def bm3d_step2(
    noisy_image,
    estimated_image,
    sigma,
    tau,
    window_size=5,
    search_window=11,
):

    padded_image = np.pad(
        np.asarray(noisy_image),
        (
            (window_size // 2 + 1, window_size // 2 + 1),
            (window_size // 2 + 1, window_size // 2 + 1),
            (0, 0),
        ),
        "reflect",
    )

    padded_estimate = np.pad(
        np.asarray(estimated_image),
        (
            (window_size // 2 + 1, window_size // 2 + 1),
            (window_size // 2 + 1, window_size // 2 + 1),
            (0, 0),
        ),
        "reflect",
    )

    n_row, n_col = noisy_image.shape[0], noisy_image.shape[1]
    with ProgressBar(total=n_row * n_col) as progress_bar:
        denoised_image = _wiener_main_loop(
            progress_bar,
            noisy_image,
            padded_image,
            padded_estimate,
            sigma,
            tau,
            window_size,
            search_window,
        )

    return denoised_image
