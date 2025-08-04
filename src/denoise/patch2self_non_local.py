import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar
from numpy.typing import ArrayLike, NDArray


from .patch_denoise import wavelet_denoise_3d, fft4, low_pass_filter4d

@njit(nogil=True, fastmath=True)
def search_windows_llr_hd_3d(
    local_patch: NDArray,
    padded_image: NDArray,
    window_size: int,
    i_start: int,
    i_end: int,
    j_start: int,
    j_end: int,
    k_start: int,
    k_end: int,
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
            low_pass_filter4d(fft4(local_patch.astype("complex64")), lambda2d)
            / window_size**3
            # hard_thresholding(fft3(local_patch), lambda2d) / window_size**2
        )
        # print("local_patch", local_patch.shape, local_patch.dtype)
    elif patch_transform == "wavelet":
        local_patch_transformed = wavelet_denoise_3d(
            local_patch.astype("complex64"), threshold=lambda2d, axis=(0, 1, 2)
        )
    else:
        local_patch_transformed = local_patch.astype("complex64")
    ii = 0
    dist_list = np.zeros(
        (i_end - i_start) * (j_end - j_start) * (k_end - k_start), dtype=np.float32
    )
    for k in range(k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                # if True or not (i == ((i_end - i_start) // 2) and j == ((j_end - j_start) // 2)):
                patch = padded_image[
                    k : k + window_size, i : i + window_size, j : j + window_size
                ]
                if patch_transform == "fft":
                    # patch = hard_thresholding(wavelet_transform(patch)[1], lambda2d)
                    patch_transformed = (
                        low_pass_filter4d(fft4(patch.astype("complex64")), lambda2d)
                        / window_size**3
                        # hard_thresholding(fft3(patch), lambda2d) / window_size**2
                    )
                elif patch_transform == "wavelet":
                    patch_transformed = wavelet_denoise_3d(
                        patch.astype("complex64"), threshold=lambda2d, axis=(0, 1, 2)
                    )
                else:
                    patch_transformed = patch.astype("complex64")
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

@njit(nogil=True, fastmath=True)
def p2s(patches, alpha=1e-6, method="lstsq"):
    """
    Patch2Self denoising algorithm.
    This function applies the Patch2Self algorithm to denoise patches of an image.
    Args:
        patches: A 4D numpy array of shape (num_patches, patch_height, patch_width, num_channels).
        alpha: Tikhonov regularization parameter.
    Returns:
        output_patches: A 4D numpy array of denoised patches.
    """
    patches_shape = patches.shape
    patches = patches.reshape(patches_shape[0], -1)
    patches = np.ascontiguousarray(patches.T.conj())
    output_patches = np.empty_like(patches)
    # print("Patches shape", patches.shape, "dtype", patches.dtype)
    for j in range(patches.shape[0]):
        test_patch = patches[j, :]
        # mean_y = 0.0 * np.mean(test_patch)
        mean_y = np.mean(test_patch)
        test_patch -= mean_y
        training_patches = patches[np.arange(patches.shape[0]) != j, :]
        training_patches = np.ascontiguousarray(training_patches.T.conj())
        # mean_x = (
        #     0.0
        #     * np.array(
        #         [
        #             training_patches[:, i].mean()
        #             for i in range(training_patches.shape[1])
        #         ],
        #         dtype=training_patches.dtype,
        #     )
        # ).astype(training_patches.dtype)
        mean_x = (np.array([training_patches[:, i].mean() for i in range(training_patches.shape[1])], dtype=training_patches.dtype)).astype(training_patches.dtype)
        training_patches -= mean_x[np.newaxis, :]

        if method == "lstsq":
            w = np.linalg.lstsq(
                training_patches.T.conj() @ training_patches
                + np.float32(alpha)
                * np.eye(training_patches.shape[1], dtype=training_patches.dtype),
                training_patches.T.conj() @ test_patch,
            )[0]
        elif method == "svd":
            U, S, Vh = np.linalg.svd(training_patches, full_matrices=False)
            w = np.dot(
                Vh.T.conj(),
                np.dot(
                    np.diag(S / (S**2 + alpha)).astype(patches.dtype),
                    # np.dot(U.T.conj(), test_patch),
                    U.T.conj() @ test_patch,
                ),
            )
        elif method == "pinv":
            w = (
                np.linalg.inv(
                    training_patches.T.conj() @ training_patches
                    + np.float32(alpha)
                    * np.eye(training_patches.shape[1], dtype=training_patches.dtype)
                )
                @ training_patches.T.conj()
                @ test_patch
            )

        # output_patches[j, test_patch.shape[0] // 2] = np.dot(training_patches, w)[test_patch.shape[0] // 2]
        # output_patches[j, test_patch.shape[0] // 2] += - np.dot(mean_x, w)
        # output_patches[j, test_patch.shape[0] // 2] += mean_y
        output_patches[j, :] = np.dot(training_patches, w)
        output_patches[j, :] += - np.dot(mean_x, w)
        output_patches[j, :] += mean_y
    return output_patches.T.reshape(patches_shape)


@njit(nogil=True, parallel=True)
def _main_loop_non_local_p2s(
    progress_bar: ProgressBar,
    alpha: float,
    noisy_image: NDArray,
    padded_image: NDArray,
    window_size: int,
    patch_transform: str,
    lambda2d: float,
    search_window: int,
    tau: float,
) -> NDArray:
    """
    Main loop for the non-local patch2self denoising algorithm.
    This function processes each patch in the noisy image, searches for similar patches
    in the padded image, applies the patch2self method to denoise them, and accumulates
    the results to produce a denoised output image.

    Args:
        progress_bar: A progress bar object to track the loop's progress.
        alpha: Tikhonov regularization parameter for the patch2self method. Default is 1e-6.
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
        # print("Processing patch", t, "/", total)
        # slice_index, row, col = np.unravel_index(t, (n_slices, n_row, n_col))
        slice_index = t // (n_row * n_col)
        row = (t // n_col) % n_row
        col = t % n_col
        slice_index = int(slice_index)
        row = int(row)
        col = int(col)

        # print(t, "Processing slice", slice_index, "row", row, "col", col)

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
        # print(
        #     "Local patch shape:",
        #     local_patch.shape,
        #     "dtype:",
        #     local_patch.dtype,
        #     "k_start:",
        #     k_start,
        #     "k_end:",
        #     k_end,
        #     "i_start:",
        #     i_start,
        #     "i_end:",
        #     i_end,
        #     "j_start:",
        #     j_start,
        #     "j_end:",
        #     j_end,
        # )

        patches, indicies = search_windows_llr_hd_3d(
            local_patch,
            padded_image,
            window_size,
            i_start,
            i_end,
            j_start,
            j_end,
            k_start,
            k_end,
            patch_transform,
            lambda2d,
            tau,
        )

        patches = np.ascontiguousarray(np.transpose(patches, (1, 2, 3, 0, 4)))
        patches = patches.reshape(
            window_size * window_size * window_size, indicies.shape[0], n_channels
        )

        # patches = np.lib.stride_tricks.sliding_window_view(
        #     padded_image, (window_size, window_size, window_size, n_channels)
        # )

        # patches = patches.reshape(-1, window_size**3, n_channels)
        # patches = np.transpose(patches, (2, 1, 0))


        patches_rec = p2s(patches, alpha=alpha, method="svd")
        N = indicies.shape[0]
        patches_rec = np.ascontiguousarray(
            patches_rec.reshape(
                window_size, window_size, window_size, indicies.shape[0], n_channels
            )
        )
        patches_rec = np.ascontiguousarray(np.transpose(patches_rec, (3, 0, 1, 2, 4)))
        weights = 1 / N
        patches = patches_rec * weights
        for (kk, ii, jj), patch in zip(indicies, patches):
            numerator[
                kk : kk + window_size, ii : ii + window_size, jj : jj + window_size
            ] += patch
            denominator[
                kk : kk + window_size, ii : ii + window_size, jj : jj + window_size
            ] += weights

        progress_bar.update(1)

    denoised = numerator / (denominator + 1e-10)
    return denoised[
        window_size // 2 + 1 : -window_size // 2,
        window_size // 2 + 1 : -window_size // 2,
        window_size // 2 + 1 : -window_size // 2,
    ]


def non_local_patch2self(
    noisy_image: ArrayLike,
    alpha: float = 1e-6,
    tau: float = 0.1,
    patch_transform: str = "fft",
    lambda2d: float = 1,
    window_size: int = 5,
    search_window: int = 11,
) -> NDArray:
    """
    Perform denoising on a noisy image using the local patch2self decomposition method.

    Args:
        noisy_image: The input noisy image as a 3D NumPy array (height, width, channels).
        alpha: Tikhonov regularization parameter for the patch2self method. Default is 1e-6.
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

    assert noisy_image.ndim == 4, "Input image must be a 4D array (slices, height, width, channels)."
    noisy_image = np.asarray(noisy_image, dtype=np.float32)
    padded_image = np.pad(
        np.asarray(noisy_image),
        (
            (window_size // 2 + 1, window_size // 2 + 1),
            (window_size // 2 + 1, window_size // 2 + 1),
            (window_size // 2 + 1, window_size // 2 + 1),
            (0, 0),
        ),
        "reflect",
    )
    print("Padded image shape:", padded_image.shape, "dtype:", padded_image.dtype)
    n_slices, n_row, n_col = (
        noisy_image.shape[0],
        noisy_image.shape[1],
        noisy_image.shape[2],
    )
    with ProgressBar(total=n_slices * n_row * n_col) as progress_bar:
        denoised_image = _main_loop_non_local_p2s(
            progress_bar,
            alpha,
            noisy_image,
            padded_image,
            window_size,
            patch_transform,
            lambda2d,
            search_window,
            tau,
        )
    return denoised_image
