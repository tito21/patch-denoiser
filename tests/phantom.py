import random

import numpy as np

def ellipse(X, Y, a, b, x0, y0, theta):
    X = X - x0
    Y = Y - y0
    # fmt: off
    return (X * np.cos(theta) + Y * np.sin(theta)) ** 2 / a**2 + \
           (X * np.sin(theta) - Y * np.cos(theta)) ** 2 / b**2 <= 1
    # fmt: on


def gauss(X, Y, x0, y0, sigma):
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))


def texture(X, Y, f, x0, y0, theta):
    X = X - x0
    Y = Y - y0
    return np.sin(2 * np.pi * f *(X * np.cos(theta) + Y * np.sin(theta)))

def phantom(
    size=256, ellipses_properties=None, ellipses_contrast=None, gauss_properties=None, texture_properties=None
):
    X, Y = np.meshgrid(np.arange(size), np.arange(size))
    mag = np.zeros((size, size))
    phase = np.zeros((size, size))
    for ellipse_properties, ellipse_contrast, texture_prop in zip(
        ellipses_properties, ellipses_contrast, texture_properties
    ):
        t = ellipse_contrast * texture(X, Y, **texture_prop)
        e = ellipse(X, Y, **ellipse_properties)
        mag[e] = t[e]
        # mag[e] = ellipse_contrast

    phase = np.sum([random.random() * gauss(X, Y, **gauss_prop) for gauss_prop in gauss_properties], axis=0)
    phase = (phase - np.min(phase)) / (np.max(phase) - np.min(phase))
    return mag  * np.exp(2j * np.pi * phase)


def phantom_multicontrast(size=256, n_contrast=10, n_ellipses=15, n_gauss=2):
    ellipses_properties = [
        {
            "a": size / 2 * random.random(),
            "b": size / 2 * random.random(),
            "x0": size * random.random(),
            "y0": size * random.random(),
            "theta": random.random() * np.pi,
        }
        for _ in range(n_ellipses)
    ]

    texture_properties = [
        {
            "f": 1 / size * 2 * random.random(),
            "x0": size * random.random(),
            "y0": size * random.random(),
            "theta": random.random() * np.pi,
        }
        for _ in range(n_ellipses)
    ]

    gauss_properties = [{
        "x0": size * random.random(),
        "y0": size * random.random(),
        "sigma": size * random.random(),
    } for _ in range(n_gauss)]

    p = np.zeros((size, size, n_contrast), dtype=complex)
    for i in range(n_contrast):
        ellipses_contrast = [random.random() for _ in range(n_ellipses)]
        p[:, :, i] = phantom(
            size, ellipses_properties, ellipses_contrast, gauss_properties, texture_properties
        )
    return p
