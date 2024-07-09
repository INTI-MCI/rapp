import logging

import numpy as np

from rapp import constants as ct
from rapp.simulations import simulation
from rapp.analysis.plot import Plot

logger = logging.getLogger(__name__)

TPL_LOG = "samples={}, φerr: {}."
TPL_LABEL = "cycles={}\nstep={}°\nreps={}"
TPL_FILENAME = "sim_error_vs_samples-reps-{}-cycles-{}-step-{}.png"


def run(
    folder,
    angle=22.5,
    method="NLS",
    samples=None,
    step=1,
    reps=1,
    cycles=1,
    k=0,
    dynamic_range=0.7,
    max_v=4.096,
    show=False,
    save=True,
):
    print("")
    logger.info("PHASE DIFFERENCE VS SAMPLES")

    # n_samples that whose measurement time is integer multiple of the period of 50Hz.
    n_samples = (np.arange(10, 70, step=10) * (1 / 50) * 845).astype(int)
    amplitude = (max_v * dynamic_range) / 2

    errors = {}
    for method in simulation.METHODS:
        logger.info("Method: {}, cycles={}, step={}, reps={}".format(method, cycles, step, reps))

        errors[method] = []
        for samples in n_samples:
            n_results = simulation.n_simulations(
                N=reps,
                angle=angle,
                method=method,
                allow_nan=True,
                cycles=cycles,
                step=step,
                samples=samples,
                A=amplitude,
                max_v=max_v,
                a0_k=k,
            )

            error = n_results.rmse()
            errors[method].append(error)

            logger.info(TPL_LOG.format(samples, "{:.2E}".format(error)))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.SAMPLES_PER_ANGLE, ysci=True, xint=False, folder=folder
    )

    for method, plot_config in simulation.METHODS.items():
        plot.add_data(n_samples, errors[method], label=method, **plot_config)

    annotation = TPL_LABEL.format(cycles, step, reps)
    plot._ax.text(0.05, 0.05, annotation, transform=plot._ax.transAxes)

    plot.legend(fontsize=12)

    if save:
        plot.save(filename=TPL_FILENAME.format(reps, cycles, step))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return n_samples, errors
