import logging

import numpy as np

from rapp import constants as ct
from rapp.adc import ADC_BITS, ADC_MAXV
from rapp.simulations import simulation
from rapp.analysis.plot import Plot

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
    cycles=4,
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

    cycles_list = np.arange(1, cycles + 1, step=1)

    BITS = [ARDUINO_BITS, ADC_BITS]
    MAXV = [ARDUINO_MAXV, ADC_MAXV]
    MS = ["o", "s"]
    LS = ["dotted", "solid"]

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

    for bits, ms, ls in zip(BITS, MS, LS):
        plot.add_data(
            cycles_list,
            errors_per_bits[bits],
            ms,
            ls=ls,
            lw=2,
            mfc="None",
            mew=2,
            color="k",
            label="bits={}.".format(bits),
        )

    annotation = TPL_LABEL.format(cycles, step, samples, reps)
    plot._ax.text(0.05, 0.4, annotation, transform=plot._ax.transAxes)
    plot._ax.set_yscale("log")
    plot.legend(fontsize=12)

    if save:
        for format_ in simulation.FORMATS:
            plot.save(filename=TPL_FILENAME.format(reps, cycles, step, samples, format_))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return cycles_list, errors_per_bits
