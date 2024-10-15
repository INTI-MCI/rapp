import logging
from fractions import Fraction

import numpy as np

from rapp import adc
from rapp import constants as ct
from rapp.analysis.plot import Plot
from rapp.simulations import simulation

logger = logging.getLogger(__name__)


TPL_LOG = "step={}, reps={}, φerr: {}, mean u: {}."
TPL_LABEL = "cycles={}\nsamples={}"
TPL_FILENAME = "sim_error_vs_step--cycles-{}-samples-{}.{}"

STEPS = [0.1, 0.5, 1, 1.5, 2, 2.5]
MREPS = [10, 20, 50, 100, 200, 500]

MAX_V = 4.096


def run(
    folder,
    angle=None,
    cycles=1,
    samples=5,
    steps=STEPS,
    mreps=MREPS,
    reps=None,
    k=0,
    max_v=adc.MAXV,
    dynamic_range=0.7,
    show=False,
    save=True,
):
    print("")
    logger.info("PHASE DIFFERENCE VS STEP")

    amplitude = (max_v * dynamic_range) / 2

    if angle is None:
        angle = np.random.uniform(low=0, high=0.5, size=reps)

    if reps is not None:
        mreps = [reps for _ in steps]

    errors = {}
    for method in simulation.METHODS:
        logger.info("Method: {}, cycles={}, samples={}".format(method, cycles, samples))

        errors[method] = []
        for step, reps in zip(steps, mreps):
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

            logger.info(TPL_LOG.format(step, reps, "{:.2E}".format(error), n_res.mean_u()))

    plot = Plot(ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_STEP, ysci=True, xint=True, folder=folder)

    for method, plot_config in simulation.METHODS.items():
        plot.add_data(steps, errors[method], label=method, **plot_config)

    annotation = TPL_LABEL.format(cycles, samples)
    xticks = np.array([str(Fraction(s).limit_denominator()) for s in steps])
    yfmt = simulation.get_axis_formatter(power_limits=(-3, -3))

    plot.the_ax.text(0.05, 0.85, annotation, transform=plot.the_ax.transAxes)
    plot.the_ax.yaxis.set_major_formatter(yfmt)
    plot.the_ax.set_xscale("log", base=2)
    plot.the_ax.set_xticks(steps, xticks)

    plot.legend(loc="lower right", fontsize=12)

    if save:
        for format_ in simulation.FORMATS:
            plot.save(filename=TPL_FILENAME.format(cycles, samples, format_))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return steps, errors
