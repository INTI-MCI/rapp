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
    all_positive: bool = False,
    k: int = 0,
    angle_accuracy: float = 0,
    angle_precision: float = 0
) -> tuple:
    """Generates a one-dimensional discrete harmonic signal.
        Allows to add noise and voltage quantization.

    Args:
        A: amplitude (peak) of the signal.
        cycles: number of cycles.
        fc: samples per cycle.
        samples: samples per angle.
        phi: phase (radians).
        noise: (mu, sigma) sigma of additive white Gaussian noise that depends on the intensity mu.
        bits: number of bits for quantization. If None, doesn't quantize the signal.
        max_v: maximum value of ADC scale [0, max_v] (in Volts).
        all_positive: if true, shifts the signal to the positive axis.
        k: amount of distortion to add.
        angle_accuracy: peak to peak deviation of requested angle positions. (radians)
        angle_precision: standard deviation of angle around requested value. (radians)

    Returns:
        The signal as an (xs, ys) tuple.
    """
    def normalize_sigma(mu, sigma):
        if mu == 0:
            mu = 1

        sigma = sigma / mu

        return sigma

    n_angles = int(cycles * fc)
    xs = np.linspace(0, 2 * np.pi * cycles, num=n_angles, endpoint=False)

    if angle_accuracy:
        xs_noisy = xs + np.random.uniform(-angle_accuracy, angle_accuracy, n_angles)
    else:
        xs_noisy = xs

    if angle_precision:
        xs_noisy = xs_noisy + angle_precision * np.random.randn(n_angles)

    # All samples are taken from the reached position
    xs_noisy = np.repeat(xs_noisy, samples)
    xs = np.repeat(xs, samples)

    signal = A * np.sin(xs_noisy + phi)

    additive_noise = np.zeros(xs.size)
    if noise is not None:
        sigma = normalize_sigma(*noise)
        additive_noise = np.random.normal(loc=0, scale=sigma, size=xs.size) * signal

    signal = signal + additive_noise

    if bits is not None:
        signal = quantize(signal, max_v=max_v, bits=bits)

    if k >= 1:
        raise ValueError("distortion level k must be between 0 and 1.")

    if k > 0:
        negative = signal < 0
        signal[negative] = (1 / k) * np.arctan(k * signal[negative])
        positive = signal >= 0
        signal[positive] = signal[positive] + k * signal[positive] ** 2

    if all_positive:
        signal = signal + A

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

    return q_signal * q_factor
