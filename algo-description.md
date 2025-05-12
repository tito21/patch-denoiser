# Algo 0: Full denoising

Denoise multichannel image by low rank in a higher order tensor. It is heavily inspired by BM3D and HD-PROST. The main
idea is to search for similar patches in image space and stacking them in a $N_p \times N_c \times N$ tensor. This tensor is then
compress by hard thresholding the higher order singular values from a Tucker decomposition. After reconstructing the
low rank approximation of the tensor the patches are positioned back into the image.

 1. Pad input image by `window_size // 2 + 1` in the row and column dimension
 2. Initialize two empty arrays, `denominator` and `numerator` with size of the padded image
 3. Call `search_windows_llr_hd` with the start and end indices. This will select the similar patches to the current
 window
 4. Call `compress_tucker` to do a Tucker decomposition of the collection of patches and compress them by doing a hard
 thresholding
 5. Weight the patches by the number elements grater than zero in the core of the Tucker decomposition
 6. Add each patch to the `numerator` and in the weights to the `denominator`

## Subroutines

### Algo 1: `search_windows_llr_hd`

Searches for the must similar patches.The search is done in a denoised patch. Returns an
array of patches and of selected indices. It currently supports 3 denoising algorithms: low pass fft filter, hard
thresholding of wavelet coefficients and identity (no denoising)

 1. Apply a simple denoising algorithm on the input patch
 2. Iterate for each patch in the search window
 3. Denoise the current patch with the simple denoising algorithm
 4. Calculate the distance to the input patch as the norm of the difference scaled by the size of the window
 5. If the distance is less than the threshold add the patch and current index to the output arrays

### Algo 2: `compress_tucker`

Performs a Tucker decomposition (extension of the SVD to higher dimensions) on the collection of patches and then
compresses the tensor by doing a hard threshold on the core (eliminates small singular values). It also returns the
number of elements in the core that are grater then zero

 1. Perform the Tucker decomposition
 2. On the core tensor perform hard thresholding
 3. Reconstruct the tensor with the new core

### Algo 3: `hard_thresholding`

Set to zero all elements that are less of a threshold

