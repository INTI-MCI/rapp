import logging

from rapp.simulations import simulator
from rapp.measurement import Measurement
from rapp.analysis.phase_diff import plot_phase_difference

logger = logging.getLogger(__name__)

TPL_LABEL = "step={}° deg.\ncycles={}.\nφ = {}°.\n"
TPL_FILENAME = "sim_phase_diff_fit-samples-{}-step-{}.png"


def run(phi, folder, method='ODR', samples=50, step=1, cycles=2, reps=None, show=False):
    print("")
    logger.info("SIGNAL AND PHASE DIFFERENCE SIMULATION")

    fc = simulator.samples_per_cycle(step=step)

    logger.info("Simulating signals...")
    measurement = Measurement.simulate(
        cycles=cycles,
        fc=fc,
        phi=phi,
        fa=samples,
    )

    filename = 'sim-one-phase_diff.png'

    plot_phase_difference(measurement, method, filename, show=show)
