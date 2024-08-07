import logging

import numpy as np
import matplotlib.pyplot as plt

from rapp import adc
from rapp import constants as ct
from rapp.analysis.plot import Plot
from rapp.simulations import simulation


logger = logging.getLogger(__name__)

TPL_LOG = "cycles={}, φerr: {}."
TPL_LABEL = "step={}°\nsamples={}\nreps={}"
TPL_FILENAME = "sim_error_vs_cycles-reps-{}-step-{}-samples-{}.{}"


def run(
    folder,
    angle=None,
    samples=5,
    step=1,
    reps=1,
    cycles=4,
    k=0,
    dynamic_range=0.7,
    max_v=adc.MAXV,
    show=False,
    save=True,
):
    print("")
    logger.info("PHASE DIFFERENCE VS # OF CYCLES")

    cycles_list = np.arange(0.5, cycles + 0.5, step=0.5)
    amplitude = (max_v * dynamic_range) / 2

    if angle is None:
        angle = np.random.uniform(low=0, high=0.5, size=reps)

    logger.info("Angles simulated: {}".format(angle))

    errors = {}
    for method in simulation.METHODS:
        logger.info("Method: {}, reps={}".format(method, reps))

        errors[method] = []
        for cycles in cycles_list:
            n_res = simulation.n_simulations(
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

            error = n_res.rmse()
            errors[method].append(error)

            logger.info(TPL_LOG.format(cycles, "{:.2E}".format(error)))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_N_CYCLES, ysci=True, xint=True, folder=folder
    )

    for method, plot_config in simulation.METHODS.items():
        plot.add_data(cycles_list, errors[method], label=method, **plot_config)

    plot.legend(loc="center right", fontsize=12)

    annotation = TPL_LABEL.format(step, samples, reps)
    plot._ax.text(0.05, 0.05, annotation, transform=plot._ax.transAxes)
    yfmt = simulation.get_axis_formatter(power_limits=(-3, -3))
    plot._ax.yaxis.set_major_formatter(yfmt)
    plot._ax.yaxis.set_major_locator(plt.MaxNLocator(2))

    if save:
        for format_ in simulation.FORMATS:
            plot.save(filename=TPL_FILENAME.format(reps, step, samples, format_))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return cycles_list, errors
