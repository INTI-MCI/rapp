import logging

from rapp import constants as ct
from rapp.simulations import simulation
from rapp.analysis.plot import Plot

logger = logging.getLogger(__name__)


TPL_LOG = "step={}, reps={}, φerr: {}, mean u: {}."
TPL_LABEL = "cycles={}\nsamples={}"
TPL_FILENAME = "sim_error_vs_step--cycles-{}-samples-{}.{}"

STEPS = [0.001, 0.01, 0.1, 1, 2, 4]
MREPS = [1, 20, 50, 100, 200, 500]

MAX_V = 4.096


def run(
    folder,
    angle=22.5,
    cycles=2,
    samples=5,
    steps=STEPS,
    mreps=MREPS,
    reps=None,
    k=0,
    max_v=4.096,
    dynamic_range=0.7,
    show=False,
    save=True,
):
    print("")
    logger.info("PHASE DIFFERENCE VS STEP")

    amplitude = (max_v * dynamic_range) / 2

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
    plot._ax.text(0.05, 0.75, annotation, transform=plot._ax.transAxes)

    plot._ax.set_xscale("log")
    plot.legend(loc="lower right", fontsize=12)

    if save:
        for format_ in simulation.FORMATS:
            plot.save(filename=TPL_FILENAME.format(cycles, samples, format_))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return steps, errors
