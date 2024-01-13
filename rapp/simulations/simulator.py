import os
import logging

import numpy as np

from rapp import constants as ct
from rapp.measurement import Measurement
from rapp.utils import create_folder, round_to_n

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


PHI = np.pi / 8         # Phase difference.

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

    def mean_u(self, degrees=False):
        return round_to_n(np.rad2deg(np.mean(self._us)), 2)


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
        *head, res = measurement.phase_diff(
            method=method, p0=p0, allow_nan=allow_nan, degrees=False
        )
        results.append(res)

    return SimulationResult(phi, results)


def main(sim, phi=PHI, method='ODR', reps=1, step=1, samples=50, show=False):
    print("")
    logger.info("STARTING SIMULATIONS...")

    logger.info("PHASE DIFFERENCE: {} degrees.".format(np.rad2deg(PHI)))

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
