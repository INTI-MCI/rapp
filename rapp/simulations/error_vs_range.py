import logging

import numpy as np

from rapp import constants as ct
from rapp.adc import GAINS, GAIN_ONE
from rapp.simulations import simulation
from rapp.analysis.plot import Plot


logger = logging.getLogger(__name__)

TPL_LOG = "A={}, φerr: {}."
TPL_LABEL = "cycles={}\nsamples={}\nstep={}°\nreps={}"
TPL_FILENAME = "sim_error_vs_range-reps-{}-cycles-{}-samples-{}-step-{}.png"

np.random.seed(1)


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


def run(
    folder, angle=22.5,
    method='NLS', samples=5, step=1, reps=1, cycles=1, k=0, show=False, save=True
):
    print("")
    logger.info("PHASE DIFFERENCE VS MAX TENSION")

    max_v, step_mV = GAINS[GAIN_ONE]
    v_step = step_mV * 2
    max_A = max_v / 2
    amplitudes = np.linspace(max_A - v_step * 8, max_A, num=8)
    percentages = ((amplitudes * 2) / max_v) * 100

    logger.info("MAX V={}".format(max_v))

    errors = {}
    for method in METHODS:
        logger.info("Method: {}, cycles={}, step={}, reps={}".format(method, cycles, step, reps))

        errors[method] = []
        for amplitude in amplitudes:
            n_results = simulation.n_simulations(
                angle=angle,
                N=reps,
                A=amplitude,
                max_v=max_v,
                method=method,
                cycles=cycles,
                step=step,
                samples=samples,
                allow_nan=True,
                a0_k=k
            )

            error = n_results.rmse()
            errors[method].append(error)

            logger.info(TPL_LOG.format(amplitude, "{:.2E}".format(error)))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR,
        xlabel=ct.LABEL_DYNAMIC_RANGE_USE,
        ysci=True,
        xint=False,
        folder=folder
    )

    for method, plot_config in METHODS.items():
        plot.add_data(percentages, errors[method], label=method, **plot_config)

    # plot._ax.set_yscale('log')
    plot.legend(loc='upper right', fontsize=12)

    annotation = TPL_LABEL.format(cycles, samples, step, reps)
    plot._ax.text(0.25, 0.7, annotation, transform=plot._ax.transAxes)

    if save:
        plot.save(filename=TPL_FILENAME.format(reps, cycles, samples, step))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return percentages, errors
