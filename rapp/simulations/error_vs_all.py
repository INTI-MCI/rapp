import logging

import numpy as np

from rapp import constants as ct
from rapp.analysis.plot import Plot
from rapp.simulations import simulation

logger = logging.getLogger(__name__)


TPL_LOG = "cycles={}, φerr: {}."
TPL_LABEL = "reps={}"
TPL_FILENAME = "sim_error_vs_all-reps-{}.{}"


def run(
    folder,
    angle=None,
    method="DFT",
    steps_samples=((1, 169)),
    reps=1,
    cycles=2,
    k=0,
    dynamic_range=0.7,
    max_v=4.096,
    show=False,
    save=True,
):
    print("")
    logger.info("PHASE DIFFERENCE VS # OF CYCLES FOR DIFFERENT PARAMETERS")

    cycles_list = np.arange(0.5, cycles + 0.5, step=0.5)
    amplitude = (max_v * dynamic_range) / 2

    if angle is None:
        angle = np.random.uniform(low=0, high=0.5, size=reps)

    errors = {}
    for step, samples in steps_samples:
        logger.info(
            "Method: {}, (step, samples)=({}, {}), reps={}".format(method, step, samples, reps)
        )

        errors[(step, samples)] = []
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
            errors[(step, samples)].append(error)

            logger.info(TPL_LOG.format(cycles, "{:.2E}".format(error)))

    plot = Plot(ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_N_CYCLES, ysci=True, folder=folder)

    for step, samples in steps_samples:
        label = f"st, sa={(step, samples)}"
        color = "k" if step == 1 and samples == 169 else None
        plot.add_data(
            cycles_list, errors[(step, samples)], style="o--", color=color, lw=1.5, label=label
        )

    annotation = TPL_LABEL.format(reps)
    plot.the_ax.text(0.05, 0.05, annotation, transform=plot.the_ax.transAxes)
    yfmt = simulation.get_axis_formatter(power_limits=(-3, -3))
    plot.the_ax.yaxis.set_major_formatter(yfmt)
    plot.the_ax.set_xticks(cycles_list)
    plot.the_ax.set_ylim(top=2.9e-3)

    plot.legend(loc="upper right", fontsize=12)

    if save:
        for format_ in simulation.FORMATS:
            plot.save(filename=TPL_FILENAME.format(reps, format_))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return cycles_list, errors
