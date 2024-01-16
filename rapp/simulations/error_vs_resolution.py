import logging

import numpy as np

from rapp import constants as ct
from rapp.adc import ADC_BITS, ADC_MAXV
from rapp.simulations import simulation
from rapp.analysis.plot import Plot

logger = logging.getLogger(__name__)

TPL_LOG = "cycles={}, φerr: {}."
TPL_LABEL = "bits={}."
TPL_FILENAME = "sim_error_vs_resolution-step-{}-samples-{}-reps{}.png"
TPL_TEXT = "step={}°\nsamples={}\nreps={}"

ARDUINO_MAXV = 5
ARDUINO_BITS = 10


def run(phi, folder, method='ODR', samples=5, step=1, reps=1, cycles=8, show=False, save=True):
    print("")
    logger.info("PHASE DIFFERENCE VS RESOLUTION")

    cycles_list = np.arange(1, cycles + 1, step=1)
    fc = simulation.samples_per_cycle(step=step)

    BITS = [ARDUINO_BITS, ADC_BITS]
    MAXV = [ARDUINO_MAXV, ADC_MAXV]
    MS = ['o', 's']
    LS = ['dotted', 'solid']

    results = []
    for bits, maxv in zip(BITS, MAXV):
        logger.info("Bits: {}".format(bits))

        amplitude = 0.9 * maxv

        errors = []
        for cycles in cycles_list:
            n_results = simulation.n_simulations(
                N=reps,
                phi=phi,
                A=amplitude,
                bits=bits,
                max_v=maxv,
                cycles=cycles,
                fc=fc,
                fa=samples,
                allow_nan=True,
                method=method,
                p0=[1, 0, 0, 0, 0, 0]
            )

            # RMSE
            error = n_results.rmse()
            errors.append(error)

            logger.info(TPL_LOG.format(cycles, "{:.2E}".format(error)))

        results.append(errors)

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR,
        xlabel=ct.LABEL_N_CYCLES,
        ysci=True,
        xint=True,
        folder=folder
    )

    for bits, errors, ms, ls in zip(BITS, results, MS, LS):
        label = TPL_LABEL.format(bits)
        plot.add_data(
            cycles_list, errors, ms, ls=ls, lw=2, mfc='None', mew=2, color='k', label=label)

    annotation = TPL_TEXT.format(step, samples, reps)
    plot._ax.text(0.61, 0.5, annotation, transform=plot._ax.transAxes)
    plot._ax.set_yscale('log')
    plot.legend(fontsize=12)

    if save:
        filename = TPL_FILENAME.format(step, samples, reps)
        plot.save(filename=filename)

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")
