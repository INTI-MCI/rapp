import logging

import numpy as np
import matplotlib.pyplot as plt

from rapp import constants as ct
from rapp.simulations import simulation
from rapp.analysis.plot import Plot

logger = logging.getLogger(__name__)


TPL_LOG = "angle={}°, φerr: {}."
TPL_LABEL = "cycles={}\nsamples={}\nstep={}°\nreps={}"
TPL_FILENAME = "sim_error_vs_phase-reps-{}-cycles-{}-samples-{}-step-{}.svg"


def run(
    folder,
    angle_range=(-90, 90),
    n_angles=30,
    cycles=4,
    step=1,
    samples=5,
    reps=1,
    k=0,
    max_v=4.096,
    dynamic_range=0.7,
    show=False,
    save=True,
):
    print("")
    logger.info("PHASE DIFFERENCE VS PHASE ANGLE")

    angles = np.linspace(*angle_range, num=n_angles)
    amplitude = (max_v * dynamic_range) / 2

    errors = {}
    for method in simulation.METHODS:
        logger.info("Method: {}, reps={}".format(method, reps))

        errors[method] = []
        for angle in angles:
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

            logger.info(TPL_LOG.format(angle, "{:.2E}".format(error)))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_ANGLE, ysci=True, xint=True, folder=folder
    )

    for method, plot_config in simulation.METHODS.items():
        plot.add_data(angles, errors[method], label=method, **plot_config)

    plot._ax.axhline(y=0, lw=2, ls="dashed", label="RMSE=0")

    annotation = TPL_LABEL.format(cycles, samples, step, reps)
    plot._ax.text(0.60, 0.1, annotation, transform=plot._ax.transAxes)

    plot.legend(loc="upper right", fontsize=13)
    plot._ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    plot._ax.set_ylim(-1e-2, 1e-2)

    if save:
        for format_ in simulation.FORMATS:
            plot.save(filename=TPL_FILENAME.format(reps, cycles, samples, step, format_))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return angles, errors
