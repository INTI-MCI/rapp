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

np.random.seed(1)  # To make random simulations repeatable.
PHI = np.pi / 8         # Phase difference.

SIMULATIONS = {
    'error_vs_cycles': error_vs_cycles,
    'error_vs_range': error_vs_range,
    'error_vs_res': error_vs_resolution,
    'error_vs_samples': error_vs_samples,
    'error_vs_step': error_vs_step,
    'one_phase_diff': one_phase_diff,
    'pvalue_vs_range': pvalue_vs_range,
    'sim_steps': simulation_steps
}


class All:
    def run(self, *args, **kwargs):
        for simulation in SIMULATIONS.values():
            simulation.run(*args, **kwargs)


def build(name):
    if name == 'all':
        return All()

    if name not in SIMULATIONS:
        raise ValueError("Simulation with name {} not implemented".format(name))

    return SIMULATIONS[name]


def samples_per_cycle(step=0.01):
    # Half cycle (180) of the analyzer is one full cycle of the signal.
    return int(180 / step)


class SimulationResult:
    def __init__(self, phi, results):
        self._phi = phi
        self._values = np.array([e.value for e in results])
        self._us = np.array([r.u for r in results])

    def rmse(self):
        true_values = np.full(shape=len(self._values), fill_value=self._phi)
        rmse = np.sqrt(np.mean(np.square(abs(true_values) - abs(self._values))))
        return np.rad2deg(rmse)

    def mean_u(self, degrees=False):
        return round_to_n(np.rad2deg(np.mean(self._us)), 2)


def n_simulations(N=1, phi=PHI, method='ODR', p0=None, allow_nan=False, **kwargs):
    """Performs N measurement simulations and calculates their phase difference.

    Args:
        N: number of simulations.
        *kwargs: arguments for Measurement.simulate method.

    Returns:
        SimulationResult: object with N phase difference results.
    """
    results = []
    for i in range(N):
        m = Measurement.simulate(phi, **kwargs)
        *head, res = m.phase_diff(method=method, p0=p0, allow_nan=allow_nan, degrees=False)
        results.append(res)

    return SimulationResult(phi, results)


def main(name, phi=PHI, method='ODR', reps=1, step=1, samples=50, cycles=2, show=False):
    print("")
    logger.info("STARTING SIMULATIONS...")
    logger.info("SIMULATED PHASE DIFFERENCE: {} degrees.".format(np.rad2deg(PHI)))

    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    simulation = build(name)
    simulation.run(
        phi,
        output_folder,
        method=method,
        samples=samples,
        step=step,
        reps=reps,
        cycles=cycles,
        show=show
    )


if __name__ == '__main__':
    main(sim='error_vs_step', show=True)
