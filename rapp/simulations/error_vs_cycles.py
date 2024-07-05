import logging

import numpy as np

from rapp import constants as ct
from rapp.simulations import simulation
from rapp.analysis.plot import Plot

logger = logging.getLogger(__name__)


METHODS = {
    'NLS': dict(
        style='s',
        ls='solid',
        lw=1.5,
        mfc=None,
        mew=1,
        color='k',
    ),
    'DFT': dict(
        style='o',
        ls='dotted',
        lw=1.5,
        mfc='None',
        mew=1.5,
        color='k',
    ),
}


TPL_LOG = "cycles={}, φerr: {}."
TPL_LABEL = "samples={}\nstep={}°\nreps={}"
TPL_FILENAME = "sim_error_vs_cycles-reps-{}-samples-{}-step-{}.png"

MAX_V = 4.096


def run(
    folder, angle=22.5, method=None, samples=5, step=1, reps=1, cycles=4, k=0,
    show=False, save=True,
):
    print("")
    logger.info("PHASE DIFFERENCE VS # OF CYCLES")

    cycles_list = np.arange(1, cycles + 1, step=1)

    errors = {}
    for method in METHODS:
        logger.info("Method: {}, reps={}".format(method, reps))

        errors[method] = []
        for cycles in cycles_list:
            n_res = simulation.n_simulations(
                angle=angle,
                N=reps,
                cycles=cycles,
                step=step,
                samples=samples,
                method=method,
                allow_nan=True,
                max_v=MAX_V,
                A=(MAX_V * 0.7) / 2,
                a0_k=k
            )

            error = n_res.rmse()
            errors[method].append(error)

            logger.info(TPL_LOG.format(cycles, "{:.2E}".format(error)))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_N_CYCLES, ysci=True, xint=True, folder=folder)

    for method, plot_config in METHODS.items():
        plot.add_data(cycles_list, errors[method], label=method, **plot_config)

    annotation = TPL_LABEL.format(samples, step, reps)
    plot._ax.text(0.05, 0.05, annotation, transform=plot._ax.transAxes)

    # plot._ax.set_yscale('log')
    plot.legend(loc='center right', fontsize=12)

    if save:
        plot.save(filename=TPL_FILENAME.format(reps, samples, step))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return cycles_list, errors
