import logging

import numpy as np

logger = logging.getLogger(__name__)


def harmonic(
    A: float = 2,
    cycles: int = 1,
    fc: int = 50,
    samples: int = 1,
    phi: float = 0,
    noise: tuple = None,
    bits: int = 16,
    max_v: float = 4,
    all_positive: bool = False
) -> tuple:
    """Generates a one-dimensional discrete harmonic signal.
        Allows to add noise and voltage quantization.

    Args:
        A: amplitude (peak) of the signal.
        cycles: number of cycles.
        fc: samples per cycle.
        samples: samples per angle.
        phi: phase (radians).
        noise: (mu, sigma) of additive white Gaussian noise.
        bits: number of bits for quantization. If None, doesn't quantize the signal.
        max_v: maximum value of ADC scale [0, max_v] (in Volts).
        all_positive: if true, shifts the signal to the positive axis.

    Returns:
        The signal as an (xs, ys) tuple.
    """

    xs = np.linspace(0, 2 * np.pi * cycles, num=int(cycles * fc), endpoint=False)
    xs = np.repeat(xs, samples)

    signal = A * np.sin(xs + phi)

    additive_noise = np.zeros(xs.size)
    if noise is not None:
        mu, sigma = noise
        additive_noise = np.random.normal(loc=mu, scale=sigma, size=xs.size)

    signal = signal + additive_noise

    if all_positive:
        signal = signal + A

    if bits is not None:
        signal = quantize(signal, max_v=max_v, bits=bits)

    return xs, signal


def quantize(
    signal: np.array, max_v: float = 4, bits: int = 16, signed=True
) -> np.array:
    """Performs quantization of a voltage signal.

    Args:
        signal: values to quantize (in Volts).
        max_v: maximum value of ADC scale [0, max_v] (in Volts).
        bits: number of bits for quantization.
        signed: if true, allows using negative values.

    Returns:
       The quantized signal.
    """
    max_q = 2 ** (bits - 1 if signed else bits)
    q_factor = max_v / max_q

    q_signal = np.round(signal / q_factor)

    q_signal[q_signal > max_q - 1] = max_q - 1
    q_signal[q_signal < -max_q] = -max_q

    logger.debug("Quantization: bits={}, factor={}".format(bits, q_factor))

    return q_signal * q_factor
