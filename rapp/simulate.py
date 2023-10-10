import numpy as np


def harmonic_signal(
    A: float = 5,
    N: int = 1,
    fs: int = 50,
    phi: float = 0,
    awgn: float = None
) -> tuple:
    """Simulates a harmonic signal.

    Args:
        A: amplitude (peak) of the signal.
        N: number of cycles.
        fs: sampling frequency (samples per cycle).
        phi: phase (radians).
        awgn: amount of additive white gaussian noise, relative to A.

    Returns:
        The signal as an (x, y) tuple.
    """

    x = np.linspace(0, 2 * np.pi * N, num=N * fs)

    signal = A * np.sin(x + phi)

    noise = np.zeros(x.size)
    if awgn is not None:
        noise = np.random.normal(scale=A * awgn, size=x.size)

    signal = signal + noise

    return x, signal
