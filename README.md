# Patch based denoising for Diffusion weighted images

This repo contains a patch based denoiser for DWI. See [here](algo-description.md) for a description of the algorithm.

To install run `pip install path/to/repo`.

```python
from patch_denoise import locally_low_rank_tucker

# data with shape (Nx, Ny, DWI)

denoised = locally_low_rank_tucker(data, 3, tau=1/sigma, lambda2d=1, patch_transform="wavelet")
```

## TODO:

 - Make the code faster
 - Support for 3D images