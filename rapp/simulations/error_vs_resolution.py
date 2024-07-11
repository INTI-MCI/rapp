import logging

import numpy as np

from rapp import adc
from rapp import constants as ct
from rapp.analysis.plot import Plot
from rapp.simulations import simulation


logger = logging.getLogger(__name__)

TPL_LOG = "cycles={}, φerr: {}."
TPL_LABEL = "cycles={}\nstep={}°\nsamples={}\nreps={}"
TPL_FILENAME = "sim_error_vs_resolution-reps-{}-cycles-{}-step-{}-samples-{}.{}"

ARDUINO_MAXV = 5
ARDUINO_BITS = 10


def run(
    folder,
    angle=22.5,
    method="DFT",
    cycles=2,
    step=1,
    samples=5,
    reps=1,
    dynamic_range=0.7,
    k=0,
    show=False,
    save=True,
):
    print("")
    logger.info("PHASE DIFFERENCE VS RESOLUTION")

    cycles_list = np.arange(0.5, cycles + 0.5, step=0.5)

    BITS = [ARDUINO_BITS, adc.BITS]
    MAXV = [ARDUINO_MAXV, adc.MAXV]

    errors_per_bits = {}
    for bits, maxv in zip(BITS, MAXV):
        logger.info("Bits: {}".format(bits))

        peak2peak = dynamic_range * maxv
        amplitude = peak2peak / 2

        errors_per_bits[bits] = []
        for cycles in cycles_list:
            n_results = simulation.n_simulations(
                N=reps,
                angle=angle,
                method=method,
                allow_nan=True,
                cycles=cycles,
                step=step,
                samples=samples,
                A=amplitude,
                max_v=maxv,
                bits=bits,
                a0_k=k,
            )

            # RMSE
            error = n_results.rmse()
            errors_per_bits[bits].append(error)

            logger.info(TPL_LOG.format(cycles, "{:.2E}".format(error)))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_N_CYCLES, ysci=True, xint=True, folder=folder
    )

    for bits, (method, plot_config) in zip(BITS, simulation.METHODS.items()):
        label = "bits={}.".format(bits)
        plot.add_data(cycles_list, errors_per_bits[bits], label=label, **plot_config)

    plot.legend(fontsize=12)

    annotation = TPL_LABEL.format(cycles, step, samples, reps)
    plot._ax.text(0.05, 0.3, annotation, transform=plot._ax.transAxes)

    if save:
        for format_ in simulation.FORMATS:
            plot.save(filename=TPL_FILENAME.format(reps, cycles, step, samples, format_))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return cycles_list, errors_per_bits
