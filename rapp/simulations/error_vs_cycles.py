import logging

import numpy as np

from rapp import constants as ct
from rapp.simulations import simulation
from rapp.analysis.plot import Plot

logger = logging.getLogger(__name__)


METHODS = {  # (marker_style, line_style, reps)
    'COSINE': ('-', 'solid', 1),
    'NLS': ('d', 'solid', None),
    # 'ODR': ('-', 'dotted', None)
}

TPL_LOG = "cycles={}, φerr: {}."
TPL_LABEL = "samples={}\nstep={}°"
TPL_FILENAME = "sim_error_vs_cycles-reps-{}-samples-{}-step-{}.png"


def run(
    folder, angle=22.5, method=None, samples=5, step=1, reps=1, cycles=4, show=False, save=True
):
    print("")
    logger.info("PHASE DIFFERENCE VS # OF CYCLES")

    cycles_list = np.arange(0.5, cycles + 0.5, step=0.5)

    errors = {}
    for method, (*head, mreps) in METHODS.items():
        if mreps is None:
            mreps = reps

        logger.info("Method: {}, reps={}".format(method, mreps))

        errors[method] = []
        for cycles in cycles_list:
            n_res = simulation.n_simulations(
                angle=angle,
                N=mreps,
                cycles=cycles,
                step=step,
                samples=samples,
                method=method,
                allow_nan=True
            )

            error = n_res.rmse()
            errors[method].append(error)

            logger.info(TPL_LOG.format(cycles, "{:.2E}".format(error)))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_N_CYCLES, ysci=True, xint=True, folder=folder)

    for method, (ms, ls, _) in METHODS.items():
        plot.add_data(
            cycles_list * 2, errors[method], style=ms, ls=ls, color='k', lw=2, label=method)

    annotation = TPL_LABEL.format(samples, step)
    plot._ax.text(0.05, 0.46, annotation, transform=plot._ax.transAxes)

    plot._ax.set_yscale('log')
    plot.legend(loc='center right', fontsize=12)

    if save:
        plot.save(filename=TPL_FILENAME.format(reps, samples, step))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return cycles_list, errors
