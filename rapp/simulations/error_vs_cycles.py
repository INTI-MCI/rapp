import logging

import numpy as np

from rapp import constants as ct
from rapp.simulations import simulator
from rapp.signal.plot import Plot

logger = logging.getLogger(__name__)


METHODS = {  # (marker_style, line_style, reps)
    'COSINE': ('-', 'solid', 1),
    'NLS': ('d', 'solid', None),
    'ODR': ('-', 'dotted', None)
}

TPL_LOG = "cycles={}, φerr: {}."
TPL_LABEL = "samples={}\nstep={}°"
TPL_FILENAME = "sim_error_vs_method-reps-{}-samples-{}-step-{}.png"


def run(phi, folder, samples=5, step=1, reps=10, max_cycles=8, show=False):
    print("")
    logger.info("PHASE DIFFERENCE VS # OF CYCLES")

    cycles_list = np.arange(1, max_cycles + 1, step=1)
    fc = simulator.samples_per_cycle(step=step)

    errors = {}
    for method, (*head, mreps) in METHODS.items():
        if mreps is None:
            mreps = reps

        logger.info("Method: {}, reps={}".format(method, mreps))

        errors[method] = []
        for cycles in cycles_list:
            n_res = simulator.n_simulations(
                phi=phi,
                N=mreps,
                cycles=cycles,
                fc=fc,
                fa=samples,
                method=method,
                p0=[1, 0, 0, 0, 0, 0]
            )

            error = n_res.rmse()
            errors[method].append(error)

            logger.info(TPL_LOG.format(cycles, "{:.2E}".format(error)))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_N_CYCLES, ysci=True, xint=True, folder=folder)

    for method, (ms, ls, _) in METHODS.items():
        plot.add_data(cycles_list, errors[method], style=ms, ls=ls, color='k', lw=2, label=method)

    annotation = TPL_LABEL.format(samples, step)
    plot._ax.text(0.05, 0.46, annotation, transform=plot._ax.transAxes)

    plot._ax.set_yscale('log')
    plot.legend(loc='center right', fontsize=12)

    plot.save(filename=TPL_FILENAME.format(reps, samples, step))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")