import logging

import numpy as np
from matplotlib.colors import Normalize

from rapp import adc
from rapp import constants as ct
from rapp.analysis.plot import Plot
from rapp.simulations import simulation


logger = logging.getLogger(__name__)

TPL_LOG = "accuracy={:.3g}, precision={:.3g}, φerr: {}."
TPL_LABEL = "cycles={}\nstep={}°\nsamples={}\nreps={}"
TPL_FILENAME = "sim_analyzer_errors-reps-{}-step-{}-samples-{}-cycles{}.{}"


def run(
    folder,
    angle=None,
    guaranteed_repeatability=0.005,
    guaranteed_accuracy=0.03,
    points_per_source=6,
    reps=1,
    samples=5,
    cycles=4,
    step=1,
    k=0,
    dynamic_range=0.7,
    max_v=adc.MAXV,
    show=False,
    save=True,
):
    print("")
    logger.info("PHASE DIFFERENCE VARIATIONS DUE TO ANALYZER ERRORS")

    angle_props_lists = [np.linspace(0, guaranteed_repeatability, points_per_source),
                         np.linspace(0, guaranteed_accuracy, points_per_source)]
    amplitude = (max_v * dynamic_range) / 2

    if angle is None:
        angle = np.random.uniform(low=0, high=0.5, size=reps)

    logger.info("Angles simulated: {}".format(angle))

    errors = {}
    for method in simulation.METHODS:
        logger.info("Method: {}, reps={}".format(method, reps))

        errors[method] = np.zeros((points_per_source, points_per_source))
        for k_p, angle_precision in enumerate(angle_props_lists[0]):
            for k_a, angle_accuracy in enumerate(angle_props_lists[1]):
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
                    angle_accuracy=angle_accuracy,
                    angle_precision=angle_precision
                )

                error = n_res.rmse()
                errors[method][k_p, k_a] = error

                logger.info(TPL_LOG.format(angle_accuracy, angle_precision,
                                           "{:.2E}".format(error)))

    plot = Plot(
        nrows=1, ncols=len(simulation.METHODS), ylabel=ct.LABEL_MOTION_REPEATABILITY,
        xlabel=ct.LABEL_MOTION_ACCURACY, ysci=False,
        xint=False, folder=folder, figsize=(20, 12)
    )

    vmin, vmax = 0, 0
    for error in errors.values():
        vmin = min(vmin, error.min())
        vmax = max(vmax, error.max())
    norm = Normalize(vmin, vmax)
    for k, (method, _) in enumerate(simulation.METHODS.items()):
        plot.add_image(angle_props_lists, errors[method], norm=norm)
        plot.set_title(method, ncol=k)
    plot.add_shared_colorbar()

    annotation = TPL_LABEL.format(cycles, step, samples, reps)
    fmt = simulation.get_axis_formatter(power_limits=(-2, 2))
    plot.set_formatter(fmt)
    plot.the_ax.text(0.05, 0.05, annotation, transform=plot.the_ax.transAxes)

    if save:
        for format_ in simulation.FORMATS:
            plot.save(filename=TPL_FILENAME.format(reps, step, samples, cycles, format_))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return angle_props_lists, errors
