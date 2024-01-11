import logging

import numpy as np

from rapp import constants as ct
from rapp.simulations import simulator
from rapp.signal.plot import Plot

logger = logging.getLogger(__name__)

TPL_LOG = "A={}, φerr: {}."
TPL_LABEL = "cycles={}\nsamples={}\nstep={}°\nreps={}"
TPL_FILENAME = "sim_error_vs_range-reps-{}-samples-{}-step-{}.png"


def run(phi, folder, method, samples=5, step=0.01, reps=1, cycles=2, show=False):
    print("")
    logger.info("PHASE DIFFERENCE VS MAX TENSION")

    xs = np.arange(0.1, 2, step=0.2)

    errors = []
    for amplitude in xs:
        fc = simulator.samples_per_cycle(step=step)

        n_results = simulator.n_simulations(
            A=amplitude,
            n=reps,
            method='ODR',
            cycles=cycles,
            fc=fc,
            phi=phi,
            fa=samples,
            a0_noise=simulator.A0_NOISE,
            a1_noise=simulator.A1_NOISE,
            bits=simulator.ADC_BITS,
            all_positive=True
        )

        error_rad = simulator.rmse(phi, [e.value for e in n_results])
        error_degrees = np.rad2deg(error_rad)
        error_degrees_sci = "{:.2E}".format(error_degrees)

        logger.info(TPL_LOG.format(amplitude, error_degrees_sci))
        errors.append(error_degrees)

    label = TPL_LABEL.format(cycles, samples, step, reps)

    percentages = ((xs * 2) / simulator.ADC_MAXV) * 100

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR,
        xlabel=ct.LABEL_DYNAMIC_RANGE_USE,
        ysci=True,
        xint=False,
        folder=folder
    )

    plot.add_data(percentages, errors, color='k', style='s-', lw=1.5, label=label)

    plot._ax.set_xticks(plot._ax.get_xticks())
    plot._ax.set_yscale('log')
    plot.legend(fontsize=12)

    plot.save(filename=TPL_FILENAME.format(reps, samples, step))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")
