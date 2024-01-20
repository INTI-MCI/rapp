import logging

from rapp.measurement import Measurement
from rapp.simulations import simulation
from rapp.analysis.phase_diff import phase_difference

logger = logging.getLogger(__name__)

TPL_LABEL = "step={}° deg.\ncycles={}.\nφ = {}°.\n"
TPL_FILENAME = "sim_phase_diff_fit-samples-{}-step-{}.png"


def run(phi, folder, method='ODR', samples=50, step=1, cycles=2, reps=None, show=False, save=True):
    print("")
    logger.info("SIGNAL AND PHASE DIFFERENCE SIMULATION")

    fc = simulation.samples_per_cycle(step=step)

    logger.info("Simulating signals...")
    measurement = Measurement.simulate(
        cycles=cycles,
        fc=fc,
        phi=phi,
        fa=samples,
    )

    filename = None
    if save:
        filename = 'sim-one-phase_diff.png'

    phase_difference(measurement, method, filename, show=show)
