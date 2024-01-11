import logging

import numpy as np
from scipy import stats

from rapp import constants as ct
from rapp.simulations import simulator
from rapp.signal.plot import Plot


logger = logging.getLogger(__name__)

TPL_LOG = "A={}, pvalue={}."
TPL_LABEL = "reps={}."
TPL_FILENAME = "sim_pvalue_vs_range-reps-{}.png"


def run(phi, folder, reps=1, show=False):
    print("")
    logger.info("PVALUE (GAUSSIAN-TEST) VS DYNAMIC RANGE")

    xs = np.arange(0.001, 0.5, step=0.05)

    mean_pvalues = []
    for A in xs:
        pvalues = []
        for rep in range(reps):
            noise = A * np.random.normal(loc=0, scale=0.00032, size=40000)
            noise = noise + max(noise)
            noise = simulator.quantize(noise, max_v=simulator.ADC_MAXV, bits=simulator.ADC_BITS)

            pvalue = stats.normaltest(noise).pvalue
            pvalues.append(pvalue)

        mean_pvalues.append(np.mean(pvalues))
        logger.info(TPL_LOG.format(round(A, 3), pvalue))

    plot = Plot(
        ylabel="p-valor",
        xlabel=ct.LABEL_DYNAMIC_RANGE_USE,
        ysci=True, xint=False,
        folder=folder
    )

    label = TPL_LABEL.format(reps)
    plot.add_data(xs, mean_pvalues, style='s-', color='k', lw=2)
    plot._ax.axhline(y=0.05, ls='--', lw=2, label="pvalue=0.5")
    plot.legend(loc='upper left', fontsize=12)

    plot._ax.text(0.05, 0.8, label, transform=plot._ax.transAxes)

    plot.save(filename=TPL_FILENAME.format(reps))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")
