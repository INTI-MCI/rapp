import numpy as np


def harmonic_signal(
    A: float = 2,
    n: int = 1,
    fc: int = 50,
    phi: float = 0,
    awgn: float = None,
    all_positive: bool = False
) -> tuple:
    """Simulates a harmonic signal measured with the differential polarizer.

    Args:
        A: amplitude (peak) of the signal.
        n: number of cycles.
        fc: samples per cycle.
        phi: phase (radians).
        awgn: amount of additive white gaussian noise, relative to A.
        all_positive: if true, shifts the signal to the positive axis.

    Returns:
        The signal as an (xs, ys) tuple.
    """

    xs = np.linspace(0, 2 * np.pi * n, num=n * fc)

    signal = A * np.sin(xs + phi)

    noise = np.zeros(xs.size)
    if awgn is not None:
        noise = np.random.normal(scale=A * awgn, size=xs.size)

    signal = signal + noise

    if all_positive:
        signal = signal + A

    xs = xs / 2

    return xs, signal
