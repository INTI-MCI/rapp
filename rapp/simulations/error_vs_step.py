import logging

from rapp import constants as ct
from rapp.simulations import simulation
from rapp.analysis.plot import Plot


logger = logging.getLogger(__name__)

TPL_LOG = "step={}, reps={}, φerr: {}, mean u: {}."
TPL_LABEL = "cycles={}\nsamples={}"
TPL_FILENAME = "sim_error_vs_step--cycles-{}-samples-{}.png"

METHODS = {
    'NLS': dict(
        style='s',
        ls='solid',
        lw=1.5,
        mfc=None,
        mew=1,
        color='k',
    ),
    'DFT': dict(
        style='o',
        ls='dotted',
        lw=1.5,
        mfc='None',
        mew=1.5,
        color='k',
    ),
}

STEPS = [0.001, 0.01, 0.1, 1, 2, 4]
MREPS = [1, 20, 50, 100, 200, 500]

MAX_V = 4.096


def run(
    folder, angle=22.5, method='NLS', samples=5, step=None, steps=STEPS, reps=1, cycles=2, k=0,
    show=False, save=True
):
    print("")
    logger.info("PHASE DIFFERENCE VS STEP")

    errors = {}
    for method in METHODS:
        logger.info(
            "Method: {}, cycles={}, samples={}".format(method, cycles, samples))

        errors[method] = []
        for step, mreps in zip(steps, MREPS):
            n_res = simulation.n_simulations(
                N=mreps,
                max_v=MAX_V,
                A=(MAX_V * 0.7) / 2,
                angle=angle,
                method=method,
                cycles=cycles,
                step=step,
                samples=samples,
                allow_nan=True,
                a0_k=k

            )

            error = n_res.rmse()
            errors[method].append(error)

            logger.info(TPL_LOG.format(step, mreps, "{:.2E}".format(error), n_res.mean_u()))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR, xlabel=ct.LABEL_STEP, ysci=True, xint=True,
        folder=folder
    )

    for method, plot_config in METHODS.items():
        plot.add_data(steps, errors[method], label=method, **plot_config)

    annotation = TPL_LABEL.format(cycles, samples)
    plot._ax.text(0.05, 0.75, annotation, transform=plot._ax.transAxes)

    plot._ax.set_xscale('log')

    plot.legend(loc="lower right", fontsize=12)

    if save:
        plot.save(filename=TPL_FILENAME.format(cycles, samples))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return steps, errors
