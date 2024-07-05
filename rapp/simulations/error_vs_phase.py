import logging

import numpy as np
import matplotlib.pyplot as plt

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


TPL_LOG = "angle={}°, φerr: {}."
TPL_LABEL = "cycles={}\nsamples={}\nstep={}°\nreps={}"
TPL_FILENAME = "sim_error_vs_phase-reps-{}-cycles-{}-samples-{}-step-{}.svg"


def run(
    folder, angle=None, method=None, samples=5, step=1, reps=1, cycles=4, k=0,
    show=False, save=True
):
    print("")
    logger.info("PHASE DIFFERENCE VS PHASE ANGLE")

    angles = np.linspace(-90, 90, num=30)

    errors = {}
    for method in METHODS:
        logger.info("Method: {}, reps={}".format(method, reps))

        errors[method] = []
        for angle in angles:
            n_res = simulation.n_simulations(
                angle=angle,
                N=reps,
                cycles=cycles,
                step=step,
                samples=samples,
                method=method,
                allow_nan=True,
                a0_k=k,
            )

            error = n_res.rmse()
            errors[method].append(error)

            logger.info(TPL_LOG.format(angle, "{:.2E}".format(error)))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_ANGLE, ysci=True, xint=True, folder=folder)

    for method, plot_config in METHODS.items():
        plot.add_data(angles, errors[method], label=method, **plot_config)

    annotation = TPL_LABEL.format(cycles, samples, step, reps)
    plot._ax.text(0.65, 0.1, annotation, transform=plot._ax.transAxes)

    # plot._ax.set_yscale('log')
    plot.legend(loc='upper right', fontsize=12)
    plot._ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    if save:
        plot.save(filename=TPL_FILENAME.format(reps, cycles, samples, step))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return angles, errors
