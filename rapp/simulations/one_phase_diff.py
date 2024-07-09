import logging

from rapp.measurement import Measurement
from rapp.analysis.phase_diff import phase_difference

logger = logging.getLogger(__name__)

TPL_LABEL = "step={}° deg.\ncycles={}.\nφ = {}°.\n"
TPL_FILENAME = "sim_phase_diff-cycles-{}-step-{}-samples-{}-k-{}.svg"


def run(
    folder,
    angle=22.5,
    method="NLS",
    cycles=1,
    step=1,
    samples=50,
    dynamic_range=0.7,
    max_v=4.096,
    k=0,
    reps=None,
    show=False,
    save=True,
):
    print("")
    logger.info("SIGNAL AND PHASE DIFFERENCE SIMULATION")
    amplitude = (max_v * dynamic_range) / 2

    logger.info("Simulating signals...")
    measurement = Measurement.simulate(
        angle=angle,
        cycles=cycles,
        step=step,
        samples=samples,
        A=amplitude,
        max_v=max_v,
        a0_k=k,
    )

    filename = TPL_FILENAME.format(cycles, step, samples, k)
    phase_difference(measurement, method, allow_nan=True, filename=filename, show=show)
