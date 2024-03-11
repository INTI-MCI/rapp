import logging

import numpy as np

from rapp import constants as ct
from rapp.adc import ADC_BITS, ADC_MAXV
from rapp.simulations import simulation
from rapp.analysis.plot import Plot

logger = logging.getLogger(__name__)

TPL_LOG = "cycles={}, φerr: {}."
TPL_LABEL = "bits={}."
TPL_FILENAME = "sim_error_vs_resolution-reps-{}-cycles-{}-step-{}-samples.svg"
TPL_TEXT = "cycles={}\nstep={}°\nsamples={}\nreps={}"

ARDUINO_MAXV = 5
ARDUINO_BITS = 10


def run(
    folder, angle=22.5, method='NLS', samples=5, step=1, reps=1, cycles=4, show=False, save=True
):
    print("")
    logger.info("PHASE DIFFERENCE VS RESOLUTION")

    cycles_list = np.arange(0.5, cycles + 0.5, step=0.5)

    BITS = [ARDUINO_BITS, ADC_BITS]
    MAXV = [ARDUINO_MAXV, ADC_MAXV]
    MS = ['o', 's']
    LS = ['dotted', 'solid']

    errors_per_bits = {}
    for bits, maxv in zip(BITS, MAXV):
        logger.info("Bits: {}".format(bits))

        peak2peak = 0.9 * maxv  # 90% of dynamic range.
        amplitude = peak2peak / 2

        errors_per_bits[bits] = []
        for cycles in cycles_list:
            n_results = simulation.n_simulations(
                N=reps,
                angle=angle,
                A=amplitude,
                bits=bits,
                max_v=maxv,
                cycles=cycles,
                step=step,
                samples=samples,
                allow_nan=True,
                method=method,
            )

            # RMSE
            error = n_results.rmse()
            errors_per_bits[bits].append(error)

            logger.info(TPL_LOG.format(cycles, "{:.2E}".format(error)))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR,
        xlabel=ct.LABEL_N_CYCLES,
        ysci=True,
        xint=True,
        folder=folder
    )

    for bits, ms, ls in zip(BITS, MS, LS):
        label = TPL_LABEL.format(bits)
        plot.add_data(
            cycles_list, errors_per_bits[bits], ms,
            ls=ls, lw=2, mfc='None', mew=2, color='k', label=label
        )

    annotation = TPL_TEXT.format(cycles, step, samples, reps)
    plot._ax.text(0.61, 0.5, annotation, transform=plot._ax.transAxes)
    plot._ax.set_yscale('log')
    plot.legend(fontsize=12)

    if save:
        filename = TPL_FILENAME.format(reps, cycles, step, samples)
        plot.save(filename=filename)

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return cycles_list, errors_per_bits
