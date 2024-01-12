import logging

import numpy as np

from rapp import constants as ct
from rapp.simulations import simulator
from rapp.signal.plot import Plot
from rapp.adc import GAINS


logger = logging.getLogger(__name__)

TPL_LOG = "A={}, φerr: {}."
TPL_LABEL = "cycles={}\nsamples={}\nstep={}°\nreps={}"
TPL_FILENAME = "sim_error_vs_range-reps-{}-samples-{}-step-{}.png"


def run(phi, folder, method, samples=5, step=0.01, reps=1, cycles=2, show=False):
    print("")
    logger.info("PHASE DIFFERENCE VS MAX TENSION")

    xs = np.arange(0.1, 2, step=0.2)

    errors = {}
    for max_v, _ in list(GAINS.values())[::2]:
        logger.info("MAX V={}, step={}".format(max_v, step))
        errors[max_v] = []
        for amplitude in xs:
            fc = simulator.samples_per_cycle(step=step)

            n_results = simulator.n_simulations(
                phi=phi,
                N=reps,
                A=amplitude,
                method='ODR',
                cycles=cycles,
                fc=fc,
                fa=samples,
            )

            error = n_results.rmse()
            errors[max_v].append(error)

            logger.info(TPL_LOG.format(amplitude, "{:.2E}".format(error)))

    percentages = ((xs * 2) / simulator.ADC_MAXV) * 100

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR,
        xlabel=ct.LABEL_DYNAMIC_RANGE_USE,
        ysci=True,
        xint=False,
        folder=folder
    )

    for max_v, errors in errors.items():
        label = "max_v={} V".format(max_v)
        plot.add_data(percentages, errors, style='.-', lw=2, label=label)

    plot._ax.set_xticks(plot._ax.get_xticks())
    plot._ax.set_yscale('log')
    plot.legend(fontsize=12)

    annotation = TPL_LABEL.format(cycles, samples, step, reps)
    plot._ax.text(0.046, 0.1, annotation, transform=plot._ax.transAxes)

    plot.save(filename=TPL_FILENAME.format(reps, samples, step))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")
