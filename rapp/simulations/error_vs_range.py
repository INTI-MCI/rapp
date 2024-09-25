import logging

import numpy as np

from rapp import adc
from rapp import constants as ct
from rapp.analysis.plot import Plot
from rapp.simulations import simulation


logger = logging.getLogger(__name__)

TPL_LOG = "A={}, φerr: {}."
TPL_LABEL = "cycles={}\nstep={}°\nsamples={}\nreps={}"
TPL_FILENAME = "sim_error_vs_range-reps-{}-cycles-{}-step-{}-samples-{}.{}"


def run(
    folder,
    angle=None,
    cycles=1,
    step=1,
    samples=5,
    reps=1,
    k=0,
    show=False,
    save=True,
):
    print("")
    logger.info("PHASE DIFFERENCE VS MAX TENSION")

    max_v, step_mV = adc.GAINS[adc.GAIN_ONE]
    voltages = np.linspace(max_v * 0.25, max_v, num=6)
    amplitudes = voltages / 2
    percentages = (voltages / max_v) * 100

    if angle is None:
        angle = np.random.uniform(low=-90, high=90, size=reps)

    logger.info("MAX V={}".format(max_v))

    errors = {}
    for method in simulation.METHODS:
        logger.info("Method: {}, cycles={}, step={}, reps={}".format(method, cycles, step, reps))

        errors[method] = []
        for amplitude in amplitudes:
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

            logger.info(TPL_LOG.format(amplitude, "{:.2E}".format(error)))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_DYNAMIC_RANGE_USE, ysci=True, folder=folder
    )

    for method, plot_config in simulation.METHODS.items():
        plot.add_data(percentages, errors[method], label=method, **plot_config)

    plot.legend(loc="upper right", fontsize=12)

    annotation = TPL_LABEL.format(cycles, step, samples, reps)
    plot.the_ax.text(0.25, 0.7, annotation, transform=plot.the_ax.transAxes)
    yfmt = simulation.get_axis_formatter(power_limits=(-3, -3))
    plot.the_ax.yaxis.set_major_formatter(yfmt)

    if save:
        for format_ in simulation.FORMATS:
            plot.save(filename=TPL_FILENAME.format(reps, cycles, step, samples, format_))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return percentages, errors
