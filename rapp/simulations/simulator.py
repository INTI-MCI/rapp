import os
import logging

import numpy as np

from rapp import constants as ct
from rapp.measurement import Measurement
from rapp.utils import create_folder

from rapp.simulations import (
    error_vs_cycles,
    error_vs_range,
    error_vs_resolution,
    error_vs_samples,
    error_vs_step,
    one_phase_diff,
    pvalue_vs_range,
    simulation_steps,
)

logger = logging.getLogger(__name__)


PHI = np.pi / 4         # Phase difference.
ANALYZER_VELOCITY = 4   # Degrees per second.

ADC_MAXV = 4.096
ADC_BITS = 16     # Use signed values with this level of quantization.

ARDUINO_MAXV = 5
ARDUINO_BITS = 10

SIMULATIONS = [
    'all',
    'error_vs_cycles',
    'error_vs_range',
    'error_vs_res',
    'error_vs_samples',
    'error_vs_step',
    'one_phase_diff',
    'pvalue_vs_range',
    'sim_steps',
]


np.random.seed(1)  # To make random simulations repeatable.


def harmonic(
    A: float = 2,
    cycles: int = 1,
    fc: int = 50,
    fa: int = 1,
    phi: float = 0,
    noise: tuple = None,
    bits: int = ADC_BITS,
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


def samples_per_cycle(step=0.01):
    # Half cycle (180) of the analyzer is one full cycle of the signal.
    return int(180 / step)


class SimulationResult:
    def __init__(self, phi, results):
        self._phi = phi
        self._results = results
        self._values = np.array([e.value for e in self._results])
        self._us = np.array([r.u for r in self._results])

    def rmse(self):
        true_values = np.full(shape=len(self._values), fill_value=self._phi)
        rmse = np.sqrt(np.mean(np.square(abs(true_values) - abs(self._values))))
        return np.rad2deg(rmse)

    def mean_u(self):
        return np.mean(self._us)


def n_simulations(N=1, phi=PHI, method='ODR', p0=None, allow_nan=False, **kwargs):
    """Performs N signal and phase difference simulations.

    Args:
        N: number of simulations.
        *kwargs: arguments for polarimeter_signal function.

    Returns:
        SimulationResult: object with N phase difference results.
    """
    results = []
    for i in range(N):
        measurement = Measurement.simulate(phi, **kwargs)
        *head, res = measurement.phase_diff(method=method, p0=p0, allow_nan=allow_nan)
        results.append(res)

    return SimulationResult(phi, results)


def main(sim, phi=PHI, method='ODR', reps=1, step=1, samples=50, show=False):
    print("")
    logger.info("STARTING SIMULATIONS...")

    logger.info("PHASE DIFFERENCE: {} degrees.".format(np.rad2deg(PHI)))
    logger.info("ANALYZER VELOCITY: {} degrees per second.".format(ANALYZER_VELOCITY))

    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    # TODO: add another subparser and split these options in different commands with parameters
    if sim not in SIMULATIONS:
        raise ValueError("Simulation with name {} not implemented".format(sim))

    if sim in ['all', 'error_vs_cycles']:
        error_vs_cycles.run(phi, output_folder, samples, step, reps, show=show)

    if sim in ['all', 'error_vs_range']:
        error_vs_range.run(phi, output_folder, method, samples, step, reps, show=show)

    if sim in ['all', 'error_vs_res']:
        error_vs_resolution.run(phi, output_folder, method, samples, step, reps, show=show)

    if sim in ['all', 'error_vs_samples']:
        error_vs_samples.run(phi, output_folder, method, step, reps, show=show)

    if sim in ['all', 'error_vs_step']:
        error_vs_step.run(phi, output_folder, method, samples, reps, show=show)

    if sim in ['all', 'pvalue_vs_range']:
        pvalue_vs_range.run(phi, output_folder, method, samples, step, reps, show=show)

    if sim in ['all', 'sim_steps']:
        simulation_steps.run(output_folder, show=show)

    if sim in ['all', 'one_phase_diff']:
        one_phase_diff.run(phi, output_folder, method, samples, step, show=show)


if __name__ == '__main__':
    main(sim='error_vs_step', show=True)
