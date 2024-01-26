import logging

import numpy as np

from rapp import constants as ct
from rapp.simulations import simulation
from rapp.analysis.plot import Plot

logger = logging.getLogger(__name__)

TPL_LOG = "samples={}, φerr: {}."
TPL_LABEL = "cycles={}\nstep={}°\nreps={}"
TPL_FILENAME = "sim_error_vs_samples-reps-{}-step-{}.png"


def run(
    angle, folder, method='NLS', samples=None, step=1, reps=1, cycles=1, show=False, save=True
):
    print("")
    logger.info("PHASE DIFFERENCE VS SAMPLES")
    logger.info("Method: {}, cycles={}, reps={}".format(method, cycles, reps))

    n_samples = np.arange(10, 200, step=20)

    errors = []
    for samples in n_samples:

        n_results = simulation.n_simulations(
            N=reps,
            angle=angle,
            A=1.7,
            method=method,
            cycles=cycles,
            step=step,
            samples=samples,
            allow_nan=True,
        )

        error = n_results.rmse()
        errors.append(error)

        logger.info(TPL_LOG.format(samples, "{:.2E}".format(error)))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.SAMPLES_PER_ANGLE, ysci=True, xint=False,
        folder=folder
    )

    label = TPL_LABEL.format(cycles, step, reps)
    plot.add_data(n_samples, errors, color='k', style='s-', lw=1.5, label=label)

    plot.legend(fontsize=12)
    plot._ax.set_yscale('log')

    if save:
        plot.save(filename=TPL_FILENAME.format(reps, step))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return n_samples, errors
