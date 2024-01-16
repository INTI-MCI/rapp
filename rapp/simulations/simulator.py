import os
import logging

import numpy as np

from rapp import constants as ct
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

np.random.seed(1)  # To make random simulations repeatable.
DEFAULT_PHI = np.pi / 8  # Phase difference.

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


def build_simulation(name):
    if name == 'all':
        return All()

    if name not in SIMULATIONS:
        raise ValueError("Simulation with name {} not implemented".format(name))

    return SIMULATIONS[name]


def main(name, phi=DEFAULT_PHI, work_dir=ct.WORK_DIR, **kwargs):
    print("")
    logger.info("STARTING SIMULATIONS...")
    logger.info("SIMULATED PHASE DIFFERENCE: {} degrees.".format(np.rad2deg(phi)))

    output_folder = os.path.join(work_dir, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    simulation = build_simulation(name)
    simulation.run(phi, output_folder, **kwargs)


if __name__ == '__main__':
    main(sim='error_vs_step', show=True)
