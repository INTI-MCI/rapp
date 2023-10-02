import numpy as np


def simulate(
    A: float = 4,
    sf: int = 57600,
    f: int = 50,
    t: int = 1,
    phi: float = 0,
    bits: int = 10,
    awgn: float = None
) -> tuple:
    """Simulates a harmonic signal.

    Args:
        A: amplitude (peak) of the signal.
        sf: sampling frequency (Hz).
        f: frequency of the signal (Hz).
        t: number of seconds to simulate.
        phi: phase (radians).
        bits: number of bits of the signal.
        awgn: amount of additive white gaussian noise, relative to A.

    Returns:
        The signal as an (x, y) tuple.
    """

    samples_per_second = int(sf / bits)
    x = np.linspace(0, 2 * np.pi * f * t, num=samples_per_second * t)

    signal = A * np.sin(x + phi)

    noise = np.zeros(x.size)
    if awgn is not None:
        noise = np.random.normal(scale=A * awgn, size=x.size)

    signal = signal + noise

    return x, signal
