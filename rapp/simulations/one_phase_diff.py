import logging

from rapp.measurement import Measurement
from rapp.analysis.phase_diff import phase_difference

logger = logging.getLogger(__name__)

TPL_LABEL = "step={}° deg.\ncycles={}.\nφ = {}°.\n"
TPL_FILENAME = "sim_phase_diff_fit-samples-{}-step-{}.png"


def run(
    folder,
    angle=22.5, method='NLS', samples=50, step=1, cycles=2, reps=None, show=False, save=True
):
    print("")
    logger.info("SIGNAL AND PHASE DIFFERENCE SIMULATION")

    logger.info("Simulating signals...")
    measurement = Measurement.simulate(
        cycles=cycles,
        step=step,
        angle=angle,
        samples=samples,
    )

    filename = None
    if save:
        filename = 'sim-one-phase_diff.png'

    phase_difference(measurement, method, filename, allow_nan=True, show=show)
