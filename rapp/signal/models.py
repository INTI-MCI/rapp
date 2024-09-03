import numpy as np


def sine(x, a, phi, c):
    return a * np.sin(x + phi) + c


def two_sines(x12, A1, A2, phi1, delta, C1, C2):
    total = len(x12)
    half = total // 2

    x1 = x12[0:half]
    x2 = x12[half:total]

    s1 = sine(x1, A1, phi1, C1)
    s2 = sine(x2, A2, phi1 + delta, C2)

    return np.hstack([s1, s2])


def two_sines_model(P, x):
    """Wrapper for ODR model."""
    return two_sines(x, *P)


def two_distorted_sines(x12, A1s, A2s, phi1s, deltas, C1, C2):
    total = len(x12)
    half = total // 2

    x1 = x12[0:half]
    x2 = x12[half:total]

    s1, s2 = np.zeros_like(x1), np.zeros_like(x2)
    for h, (A1, A2, phi1, delta) in enumerate(zip(A1s, A2s, phi1s, deltas)):
        s1 = s1 + sine((h + 1) * x1, A1, phi1, 0)
        s2 = s2 + sine((h + 1) * x2, A2, phi1 + delta, 0)
    s1 = s1 + C1
    s2 = s2 + C2

    return np.hstack([s1, s2])


def two_sines_with_harmonics(x, *p):
    n_harmonics = int((len(p) - 2) / 4)
    parts = [p[i*n_harmonics:(i+1)*n_harmonics] for i in range(4)]
    A1s, A2s, phi1s, deltas = parts
    C1, C2 = p[-2:]
    return two_distorted_sines(x, A1s, A2s, phi1s, deltas, C1, C2)


def one_distorted_sine(x12, A1s, A2s, phi1s, deltas, C1, C2):
    total = len(x12)
    half = total // 2

    x1 = x12[0:half]
    x2 = x12[half:total]

    A2 = A2s[0]
    phi1 = phi1s[0]
    delta = deltas[0]

    s1 = np.zeros_like(x1)
    for h, (A1, A2, phi1, delta) in enumerate(zip(A1s, A2s, phi1s, deltas)):
        s1 = s1 + sine((h + 1) * x1, A1, phi1, 0)
    s1 = s1 + C1

    s2 = sine(x2, A2, phi1 + delta, C2)

    return np.hstack([s1, s2])


def one_sine_with_harmonic(x, *p):
    n_harmonics = int((len(p) - 2) / 4)
    parts = [p[i*n_harmonics:(i+1)*n_harmonics] for i in range(4)]
    A1s, A2s, phi1s, deltas = parts
    C1, C2 = p[-2:]
    return one_distorted_sine(x, A1s, A2s, phi1s, deltas, C1, C2)


def two_sines_with_harmonics_objective(p, *args):
    x = args[0]
    y = args[1]
    y_sigma = args[2]
    Np = len(p)
    n_harmonics = int(Np / 6)
    parts = [p[i*n_harmonics:(i+1)*n_harmonics] for i in range(6)]
    A1s, A2s, phi1s, deltas, C1s, C2s = parts
    y_model = two_distorted_sines(x, A1s, A2s, phi1s, deltas, C1s, C2s)
    return np.sum(((y_model - y) / y_sigma) ** 2.0) / Np
