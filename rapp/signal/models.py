import numpy as np


def sine(x, a, phi, c):
    return a * np.sin(x + phi) + c


def two_sines(x12, A1, A2, phi1, delta, C1, C2):
    total = len(x12)
    half = total // 2

    x1 = x12[0:half]
    x2 = x12[half:total]

    s1 = sine(x1, A1, phi1, C1)
    s2 = sine(x2 - phi1, A2, delta, C2)

    return np.hstack([s1, s2])


def two_sines_model(P, x):
    """Wrapper for ODR model."""
    return two_sines(x, *P)
