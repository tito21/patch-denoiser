from numba import njit, prange, guvectorize
import numpy as np


SQRT_2 = np.sqrt(2)


@njit(parallel=True, fastmath=True)
def logm_explicit(tensor):
    """
    Compute the matrix logarithm of a symetric positive-definite matrix.
    Parameters
    ----------
    tensor : ndarray
        Input tensor of shape (3, 3).
    Returns
    -------
    ndarray
        Matrix logarithm of the input tensor.
    """
    original_shape = tensor.shape
    tensor = tensor.reshape(-1, 3, 3)
    result = np.zeros_like(tensor)
    for i in prange(len(tensor)):
        eigenvalues, eigenvectors = np.linalg.eigh(tensor[i])
        if np.any(eigenvalues <= 0):
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
        log_eigenvalues = np.log(eigenvalues)
        result[i] = (
            eigenvectors @ np.diag(log_eigenvalues) @ eigenvectors.T
        )
    return result.reshape(original_shape)


@njit(fastmath=True)
def expm_explicit(tensor):
    """
    Compute the matrix exponential of a symetric positive-definite matrix.
    Parameters
    ----------
    tensor : ndarray
        Input tensor of shape (3, 3).
    Returns
    -------
    ndarray
        Matrix exponential of the input tensor.
    """
    original_shape = tensor.shape
    tensor = tensor.reshape(-1, 3, 3)
    result = np.zeros_like(tensor)
    for i in prange(len(tensor)):
        eigenvalues, eigenvectors = np.linalg.eigh(tensor[i])
        exp_eigenvalues = np.exp(eigenvalues)
        result[i] = (
            eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.T
        )
    return result.reshape(original_shape)



@guvectorize(["(float64[:,:,:], float64[:,:,:])"], "(n,j,j)->(n,j,j)", nopython=True, target="parallel")
def logm(tensor, result):
    """
    Compute the matrix logarithm of a symetric positive-definite matrix.
    Parameters
    ----------
    tensor : ndarray
        Input tensor of shape (3, 3).
    Returns
    -------
    ndarray
        Matrix logarithm of the input tensor.
    """
    for i in range(len(tensor)):
        eigenvalues, eigenvectors = np.linalg.eigh(tensor[i])
        if np.any(eigenvalues <= 0):
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            Warning("Eigenvalues are negative or zero, clipping to avoid log(0)")
        log_eigenvalues = np.log(eigenvalues)
        result[i] = (
            eigenvectors @ (np.diag(log_eigenvalues) @ np.linalg.inv(eigenvectors))
        )


@guvectorize(["(float64[:,:,:], float64[:,:,:])"], "(n,j,j)->(n,j,j)", nopython=True, target="parallel")
def expm(tensor, result):
    """
    Compute the matrix exponential of a symetric positive-definite matrix.
    Parameters
    ----------
    tensor : ndarray
        Input tensor of shape (3, 3).
    Returns
    -------
    ndarray
        Matrix exponential of the input tensor.
    """
    for i in range(len(tensor)):
        eigenvalues, eigenvectors = np.linalg.eigh(tensor[i])
        exp_eigenvalues = np.exp(eigenvalues)
        result[i] = (
            eigenvectors @ (np.diag(exp_eigenvalues) @ np.linalg.inv(eigenvectors))
        )

def tensor_to_log_vector(tensor):
    """
    Convert a tensor to a vector in the log-Euclidean space.
    Formula from https://doi.org/10.1002/mrm.20965

    Parameters
    ----------
    tensor : ndarray
        Input tensor of shape (3, 3).

    Returns
    -------
    ndarray
        Vector representation of the tensor.
    """

    vector = np.zeros(tensor.shape[:-2] + (6,))
    log_tensor = logm(tensor)
    vector[..., 0] = log_tensor[..., 0, 0]
    vector[..., 1] = log_tensor[..., 1, 1]
    vector[..., 2] = log_tensor[..., 2, 2]
    vector[..., 3] = SQRT_2 * log_tensor[..., 0, 1]
    vector[..., 4] = SQRT_2 * log_tensor[..., 0, 2]
    vector[..., 5] = SQRT_2 * log_tensor[..., 1, 2]

    # vector[np.isnan(vector)] = 0
    return vector


def log_vector_to_tensor(vector):
    """
    Convert a vector in the log-Euclidean space back to a tensor.

    Parameters
    ----------
    vector : ndarray
        Input vector of shape (6,).

    Returns
    -------
    ndarray
        Tensor representation of the vector.
    """

    log_tensor = np.zeros(vector.shape[:-1] + (3, 3))
    log_tensor[..., 0, 0] = vector[..., 0]
    log_tensor[..., 1, 1] = vector[..., 1]
    log_tensor[..., 2, 2] = vector[..., 2]

    log_tensor[..., 0, 1] = vector[..., 3] / SQRT_2
    log_tensor[..., 0, 2] = vector[..., 4] / SQRT_2
    log_tensor[..., 1, 2] = vector[..., 5] / SQRT_2

    log_tensor[..., 1, 0] = vector[..., 3] / SQRT_2
    log_tensor[..., 2, 0] = vector[..., 4] / SQRT_2
    log_tensor[..., 2, 1] = vector[..., 5] / SQRT_2

    return expm(log_tensor)


@njit(fastmath=True)
def distance_patch_log_vector(tensor1, tensor2, h):
    """
    Calculate the distance between two patches of tensors in the log-Euclidan space

    Parameters
    ----------
    tensor1 : ndarray
        First tensor in log-Euclidan representation.
    tensor2 : ndarray
        Second tensor in log-Euclidan representation.
    h : float
        Bandwidth parameter.

    Returns
    -------
    float
        Distance between the two tensors.
    """
    dist_squared = np.sum((tensor1 - tensor2)**2)
    return np.exp(-dist_squared / (h**2))


@njit(parallel=True)
def _nlm_tensor_2d(
    padded_tensor, n_row, n_col, mask, patch_size=5, window_size=15, h=0.1
):


    offset = patch_size // 2
    result = np.full((n_row, n_col, 6), np.nan, dtype=np.float32)

    # result = np.zeros_like(tensor)

    total = n_row * n_col

    for t in prange(0, total):
        row = t // n_col
        col = t % n_col
        if mask[row, col] == 0:
            result[row, col, ...] = padded_tensor[row, col, ...]
        i_start = row - min(window_size, row)
        i_end = row + min(window_size + 1, n_row - row)
        j_start = col - min(window_size, col)
        j_end = col + min(window_size + 1, n_col - col)

        central_patch = padded_tensor[row : row + patch_size, col : col + patch_size, :]
        new_value = np.zeros((6,), dtype=np.float32)
        weight_sum = 0

        for i in range(i_start, i_end):
            for j in range(j_start, j_end):

                patch = padded_tensor[i : i + patch_size, j : j + patch_size, :]
                weight = distance_patch_log_vector(central_patch, patch, h)

                weight_sum += weight
                new_value += weight * patch[offset, offset, :]

        new_value /= weight_sum
        result[row, col, ...] = new_value

    return result


@njit(parallel=True)
def _nlm_tensor_3d(padded_tensor, n_slices, n_row, n_col, mask, patch_size=5, window_size=15, h=0.1):

    offset = patch_size // 2
    result = np.full((n_slices, n_row, n_col, 6), np.nan, dtype=np.float32)

    # result = np.zeros_like(tensor)

    total = n_slices * n_row * n_col

    for t in prange(0, total):
        slice_idx = t // (n_row * n_col)
        row = (t % (n_row * n_col)) // n_col
        col = t % n_col
        if mask[slice_idx, row, col] == 0:
            result[slice_idx, row, col, ...] = padded_tensor[slice_idx, row, col, ...]
        k_start = slice_idx - min(window_size, slice_idx)
        k_end = slice_idx + min(window_size + 1, n_slices - slice_idx)
        i_start = row - min(window_size, row)
        i_end = row + min(window_size + 1, n_row - row)
        j_start = col - min(window_size, col)
        j_end = col + min(window_size + 1, n_col - col)

        central_patch = padded_tensor[
            slice_idx : slice_idx + patch_size,
            row : row + patch_size,
            col : col + patch_size,
            :,
        ]
        new_value = np.zeros((6,), dtype=np.float32)
        weight_sum = 0

        for k in range(k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):

                    patch = padded_tensor[
                        k : k + patch_size, i : i + patch_size, j : j + patch_size, :
                    ]
                    weight = distance_patch_log_vector(central_patch, patch, h)

                    weight_sum += weight
                    new_value += weight * patch[offset, offset, offset, :]

        new_value /= weight_sum
        result[slice_idx, row, col, ...] = new_value


    return result


def nlm_tensor(tensor, patch_size=5, window_size=15, h=0.1, mask=None):
    """
    Apply Non-Local Means (NLM) denoising to a tensor.

    Parameters
    ----------
    tensor : ndarray
        Input tensor of shape (n_slices, n_row, n_col, 3, 3).
    patch_size : int
        Size of the patches used for NLM.
    window_size : int
        Size of the search window for NLM.
    h : float
        Bandwidth parameter for NLM.

    Returns
    -------
    ndarray
        Denoised tensor of shape (n_slices, n_row, n_col, 3, 3).
    """
    if mask is None:
        mask = np.ones(tensor.shape[:-2], dtype=np.bool_)

    tensor = np.clip(tensor, -1e8, 1e8)

    offset = patch_size // 2
    if len(tensor.shape) == 4:
        padded_tensor = np.pad(
            tensor, ((offset, offset), (offset, offset), (0, 0), (0, 0)), mode="reflect"
        )
        n_row, n_col = tensor.shape[:-2]
        log_tensor_padded = tensor_to_log_vector(padded_tensor)
        denoised_log =  _nlm_tensor_2d(log_tensor_padded, n_row, n_col, mask, patch_size, window_size, h)
        return log_vector_to_tensor(denoised_log)

    elif len(tensor.shape) == 5:
        padded_tensor = np.pad(
            tensor,
            ((offset, offset), (offset, offset), (offset, offset), (0, 0), (0, 0)),
            mode="reflect",
        )
        n_slices, n_row, n_col = tensor.shape[:-2]
        log_tensor_padded = tensor_to_log_vector(padded_tensor)
        denoised_log =  _nlm_tensor_3d(log_tensor_padded, n_slices, n_row, n_col, mask, patch_size, window_size, h)
        return log_vector_to_tensor(denoised_log)
