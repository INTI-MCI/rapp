import logging

from rapp import constants as ct
from rapp.simulations import simulation
from rapp.analysis.plot import Plot


logger = logging.getLogger(__name__)

TPL_LOG = "step={}, reps:{}, φerr: {}, mean u: {}."
TPL_LABEL = "cycles={}\nsamples={}"
TPL_FILENAME = "sim_error_vs_step-samples-{}-reps{}"

STEPS = [0.001, 0.01, 0.1, 1, 2, 4]
NREPS = [1, 10, 20, 50, 100, 200]


def run(
    angle, folder, method='NLS', samples=5, step=1, reps=1, cycles=2, show=False, save=True
):
    print("")
    logger.info("PHASE DIFFERENCE VS STEP")

    logger.info("Method: {}, cycles={}".format(method, cycles))

    steps = step
    if not isinstance(steps, list):
        steps = STEPS

    if not isinstance(reps, list):
        reps = NREPS

    errors = []
    for step, reps in zip(steps, reps):
        n_res = simulation.n_simulations(
            N=reps,
            angle=angle,
            method=method,
            cycles=cycles,
            step=step,
            samples=samples,
            allow_nan=True
        )

        error = n_res.rmse()
        errors.append(error)

        logger.info(TPL_LOG.format(step, reps, "{:.2E}".format(error), n_res.mean_u()))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_STEP, ysci=True, xint=True,
        folder=folder
    )

    label = TPL_LABEL.format(cycles, samples)

    plot.add_data(steps, errors, style='s-', mfc='k', color='k', lw=1, label=label)
    plot._ax.set_yscale('log')

    plot.legend(fontsize=12)

    if save:
        plot.save(filename=TPL_FILENAME.format(samples, reps))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return steps, errors
