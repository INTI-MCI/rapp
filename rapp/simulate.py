import numpy as np


def simulate(A: float = 4, t: int = 1, phi: float = 0, n: int = 100, awgn: float = None) -> tuple:
    """Simulates a harmonic signal.

    Args:
        A: amplitude (peak) of the signal.
        t: number of periods.
        phi: phase.
        n: number of samples.
        awgn: amount of additive white gaussian noise, relative to A.

    Returns:
        The signal as an (x, y) tuple.
    """

    x = np.linspace(0, t * 2 * np.pi, num=n)

    signal = A * np.sin(x + phi)

    noise = np.zeros(x.size)
    if awgn is not None:
        noise = np.random.normal(scale=A * awgn, size=x.size)

    signal = signal + noise

    return x, signal
