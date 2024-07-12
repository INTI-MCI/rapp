import logging

import numpy as np
import matplotlib.pyplot as plt

from rapp import constants as ct
from rapp import measurement
from rapp.simulations import simulation
from rapp.analysis.plot import Plot

logger = logging.getLogger(__name__)


TPL_LOG = "angle={}°, φerr: {}."
TPL_LABEL = "cycles={}\nstep={}°\nsamples={}\nnoise={}\nreps={}"
TPL_FILENAME = "sim_error_vs_phase-reps-{}-cycles-{}-samples-{}-step-{}.svg"


def run(
    folder,
    angle_range=(-90, 90),
    n_angles=30,
    cycles=4,
    method="DFT",
    step=1,
    samples=5,
    reps=1,
    max_v=4.096,
    dynamic_range=0.7,
    n_bits=[16, 24],
    k=0,
    noise=True,
    show=False,
    save=True,
):
    print("")
    logger.info("PHASE DIFFERENCE VS ANGLE")

    angles = np.linspace(*angle_range, num=n_angles)
    amplitude = (max_v * dynamic_range) / 2

    errors = {}
    for bits in n_bits:
        logger.info("Method: {}, bits={}, reps={}".format(method, bits, reps))

        errors[bits] = []
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
                bits=bits,
                a0_noise=measurement.A0_NOISE if noise else None,
                a1_noise=measurement.A1_NOISE if noise else None,
                a0_k=k,
            )

            error = n_res.rmse()
            errors[bits].append(error)

            logger.info(TPL_LOG.format(angle, "{:.2E}".format(error)))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR,
        xlabel=ct.LABEL_ANGLE_PLANES,
        ysci=True,
        yoom=-3,
        xint=True,
        folder=folder,
    )

    for bits, (method, plot_config) in zip(n_bits, simulation.METHODS.items()):
        plot.add_data(angles, errors[bits], label=f"bits={bits}", **plot_config)

    # plot._ax.axhline(y=0, lw=2, ls="dashed")

    annotation = TPL_LABEL.format(cycles, step, samples, noise, reps)
    plot._ax.text(0.05, 0.65, annotation, transform=plot._ax.transAxes)

    plot.legend(loc="upper right", fontsize=13)
    plot._ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    max_value = max(max(x) for x in errors.values())
    plot._ax.set_ylim(-1e-3, max_value + 0.005)

    if save:
        for format_ in simulation.FORMATS:
            plot.save(filename=TPL_FILENAME.format(reps, cycles, samples, step, format_))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return angles, errors
