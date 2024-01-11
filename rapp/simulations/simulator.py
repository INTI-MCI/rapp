import os
import logging

import numpy as np
import pandas as pd

from rapp import constants as ct
from rapp.utils import create_folder
from rapp.signal.analysis import average_data
from rapp.signal.plot import Plot
from rapp.signal.phase import phase_difference

from rapp.simulations import (
    error_vs_method,
    error_vs_step,
    error_vs_samples,
    error_vs_range,
    error_vs_resolution,
    pvalue_vs_range,
    signal_simulation
)

logger = logging.getLogger(__name__)


PHI = np.pi / 4         # Phase difference.
ANALYZER_VELOCITY = 4   # Degrees per second.

ADC_MAXV = 4.096
ADC_BITS = 16     # Use signed values with this level of quantization.

ARDUINO_MAXV = 5
ARDUINO_BITS = 10

# Noise measured with laser ON
A0_NOISE = [2.6352759502752957e-06, 0.0003747564924374617]
A1_NOISE = [3.817173720425239e-06, 0.0002145422291402638]

SIMULATIONS = [
    'all',
    'error_vs_method',
    'error_vs_step',
    'error_vs_samples',
    'error_vs_range',
    'error_vs_res',
    'sim_steps',
    'pvalue_vs_range',
    'phase_diff',
]


np.random.seed(1)  # To make random simulations repeatable.


def rmse(true_value, values):
    return np.sqrt(sum([(abs(true_value) - abs(v)) ** 2 for v in values]) / len(values))


def harmonic(
    A: float = 2,
    cycles: int = 1,
    fc: int = 50,
    fa: int = 1,
    phi: float = 0,
    noise: tuple = None,
    bits: int = None,
    max_v: float = ADC_MAXV,
    all_positive: bool = False
) -> tuple:
    """Simulates a harmonic signal.

    Args:
        A: amplitude (peak) of the signal.
        cycles: number of cycles.
        fc: samples per cycle.
        fa: samples per angle.
        phi: phase (radians).
        noise: (mu, sigma) of additive white Gaussian noise.
        bits: number of bits for quantization. If None, doesn't quantize the signal.
        max_v: maximum value of ADC scale [0, max_v] (in Volts).
        all_positive: if true, shifts the signal to the positive axis.

    Returns:
        The signal as an (xs, ys) tuple.
    """

    xs = np.linspace(0, 2 * np.pi * cycles, num=int(cycles * fc))
    xs = np.repeat(xs, fa)

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
    signal: np.array, max_v: float = ADC_MAXV, bits: int = ADC_BITS, signed=True
) -> np.array:
    """Performs quantization of a (simulated) signal.

    Args:
        signal: values to quantize (in Volts).
        max_v: maximum value of ADC scale [0, max_v] (in Volts).
        bits: number of bits for quantization.
        signed: if true, allows using negative values.

    Returns:
        Quantized signal.
    """
    max_q = 2 ** (bits - 1 if signed else bits)
    q_factor = max_v / max_q

    q_signal = np.round(signal / q_factor)

    q_signal[q_signal > max_q - 1] = max_q - 1
    q_signal[q_signal < -max_q] = -max_q

    logger.debug("Quantization: bits={}, factor={}".format(bits, q_factor))

    return q_signal * q_factor


def polarimeter_signal(phi=0, a0_noise=None, a1_noise=None, **kwargs):
    """Simulates a pair of signals measured by the polarimeter.

    Args:
        phi: phase difference between signals (radians).
        a0_noise: (mu, sigma) of additive white Gaussian noise of channel 0.
        a1_noise: (mu, sigma) of additive white Gaussian noise of channel 1.
        **kwargs: any other keyword argument to be passed 'harmonic' function.

    Returns:
        the X-coordinates and S1, S2 Y-coordinates as (X, S1, S2) 3-tuple.
    """
    xs, s1 = harmonic(noise=a0_noise, **kwargs)
    _, s2 = harmonic(phi=-phi, noise=a1_noise, **kwargs)

    # We divide angles by 2 because one cycle of the analyzer contains two cycles of the signal.
    return xs / 2, s1, s2


def samples_per_cycle(step=0.01):
    # Half cycle (180) of the analyzer is one full cycle of the signal.
    return int(180 / step)


def total_time(n_cycles):
    return n_cycles * (180 / ANALYZER_VELOCITY)


def n_simulations(n=1, method='ODR', p0=None, allow_nan=False, **kwargs):
    """Performs n simulations and returned a list of n phase difference results.

    Args:
        n: number of simulations.
        method: phase difference calculation method.
        *kwargs: arguments for polarimeter_signal function.
    """
    results = []
    for i in range(n):
        xs, s1, s2 = polarimeter_signal(**kwargs)

        data = np.array([xs, s1, s2]).T
        data = pd.DataFrame(data=data, columns=["ANGLE", "CH0", "CH1"])

        xs, s1, s2, s1_sigma, s2_sigma = average_data(data, allow_nan=allow_nan)
        x_sigma = np.deg2rad(ct.ANALYZER_UNCERTAINTY)

        if np.isnan(s1_sigma).any() or (s1_sigma == 0).any():
            s1_sigma = None

        if np.isnan(s2_sigma).any() or (s2_sigma == 0).any():
            s2_sigma = None

        res = phase_difference(
            xs * 2, s1, s2,
            x_sigma=x_sigma,
            s1_sigma=s1_sigma,
            s2_sigma=s2_sigma,
            method=method,
            degrees=False,
            p0=p0
        )

        results.append(res)

    return results


def plot_phase_diff(phi, folder, samples=50, cycles=10, step=0.01, show=False):
    print("")
    logger.info("PHASE DIFFERENCE OF TWO SIMULATED SIGNALS")

    fc = samples_per_cycle(step=step)

    logger.info("Simulating signals...")
    xs, s1, s2 = polarimeter_signal(
        cycles=cycles, fc=fc, phi=phi, fa=samples,
        a0_noise=A0_NOISE, a1_noise=A1_NOISE, all_positive=True
    )

    logger.info("Calculating phase difference...")
    res = phase_difference(xs * 2, s1, s2, method='ODR')

    error = abs(phi - res.value)
    error_degrees = np.rad2deg(error)

    logger.info("Detected phase difference: {}".format(np.rad2deg(res.value)))
    logger.info("cycles={}, fc={}, step={}°, φerr: {}.".format(cycles, fc, step, error_degrees))

    label = (
        "fc={}. \n"
        "step={}° deg. \n"
        "# cycles={}. \n"
        "|φ1 - φ2| = {}°. \n"
    ).format(fc, step, cycles, round(np.rad2deg(phi)))

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_ANGLE, folder=folder)

    markevery = int(fc / 180) if fc >= 180 else 1  # for visualization purposes, show less points.

    plot.add_data(
        xs, s1,
        ms=6, color='k', mew=0.5, xrad=True, markevery=markevery, alpha=0.8, label=label
    )

    plot.add_data(
        xs, s2,
        ms=6, color='k', mew=0.5, xrad=True, markevery=markevery, alpha=0.8
    )

    label = "φerr = {}.".format(round(error_degrees, 5))
    plot.add_data(res.fitx / 2, res.fits1, style='-', color='k', lw=1.5, xrad=True)
    plot.add_data(res.fitx / 2, res.fits2, style='-', color='k', lw=1.5, xrad=True, label=label)

    plot._ax.set_xlim(0, 1)

    plot.legend(loc='upper right')

    plot.save(filename="sim_phase_diff_fit-samples-{}-step-{}.png".format(samples, step))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")


def main(sim, method='ODR', reps=1, step=1, samples=50, show=False):
    print("")
    logger.info("STARTING SIMULATIONS...")

    logger.info("PHASE DIFFERENCE: {} degrees.".format(np.rad2deg(PHI)))
    logger.info("ANALYZER VELOCITY: {} degrees per second.".format(ANALYZER_VELOCITY))

    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    # TODO: add another subparser and split these options in different commands with parameters
    if sim not in SIMULATIONS:
        raise ValueError("Simulation with name {} not implemented".format(sim))

    if sim in ['all', 'error_vs_method']:
        error_vs_method.run(PHI, output_folder, samples, step, reps, show=show)

    if sim in ['all', 'error_vs_step']:
        error_vs_step.run(PHI, output_folder, method, samples, reps, show=show)

    if sim in ['all', 'error_vs_samples']:
        error_vs_samples.run(PHI, output_folder, method, step, reps, show=show)

    if sim in ['all', 'error_vs_range']:
        error_vs_range.run(PHI, output_folder, method, samples, step, reps, show=show)

    if sim in ['all', 'error_vs_res']:
        error_vs_resolution.run(PHI, output_folder, method, samples, step, reps, show=show)

    if sim in ['all', 'pvalue_vs_range']:
        pvalue_vs_range.run(PHI, output_folder, reps=reps, show=show)

    if sim in ['all', 'sim_steps']:
        signal_simulation.run(output_folder, show=show)

    if sim in ['all', 'phase_diff']:
        plot_phase_diff(PHI, output_folder, samples, cycles=2, step=step, show=show)


if __name__ == '__main__':
    main(sim='error_vs_step', show=True)
