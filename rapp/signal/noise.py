import os
import sys

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy import stats

import rapp.constants as ct
from rapp.signal.analysis import generate_bins
from rapp.utils import round_to_n, create_folder


def read_measurement_file(filepath, sep=r"\s+"):
    return pd.read_csv(
        filepath,
        sep=sep, skip_blank_lines=True, comment='#', usecols=(0, 1, 2), encoding=ct.ENCONDIG
    )


def poisson_fit(filepath, output_folder):
    f, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True, sharex=False)

    # base_output = "{}".format(os.path.join(output_folder, os.path.basename(filepath)[:-4]))

    data = read_measurement_file(filepath)

    for i, ax in enumerate(axs):
        print("-- CHANEL {} --".format(i))
        channel_data = data['CH{}'.format(i)].to_numpy()
        channel_data = (channel_data * (1000 / 0.125)).astype(int)  # convert to bits
        channel_data = channel_data + abs(np.min(channel_data))  # shift to positives

        mu = np.mean(channel_data)

        print("Minimum of data: {}".format(min(channel_data)))
        print("Maximum of data: {}".format(max(channel_data)))
        print("Mean of data: {}".format(mu))

        centers, edges = generate_bins(channel_data, step=1)
        print("Bins centers: {}".format(centers))
        print("Bins edges: {}".format(edges))

        counts, edges = np.histogram(channel_data, bins=edges, density=True)
        print("Sum of bar heights: {}".format(sum(counts)))

        poisson_xs = np.arange(0, max(channel_data), step=1)
        poisson_pmf = stats.poisson.pmf(poisson_xs, mu)

        mu_rounded = round_to_n(mu, 2)

        mu_label = "µ = {}.".format(mu_rounded)

        if i == 0:
            ax.set_ylabel("PMF")

        ax.set_xlabel("Bits")
        ax.set_title("Canal {}".format(i))

        # ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        # ax.xaxis.set_major_locator(plt.MaxNLocator(3))

        ax.bar(centers, counts, width=np.diff(edges), color='k', alpha=0.2, edgecolor='k')
        ax.plot(poisson_xs, poisson_pmf, color='k', lw=2, label='Poisson PMF')
        ax.axvline(x=mu, ls='--', color='k', lw=1.5, alpha=0.8, label=mu_label)

        ax.legend(loc='upper right', fontsize=10)

    plt.show()


def main(name):
    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    filename = "darkcurrent-range4V-samples100000.txt"
    filepath = os.path.join(ct.INPUT_DIR, filename)

    if name == 'poisson':
        poisson_fit(filepath, output_folder)


if __name__ == '__main__':
    main(sys.argv[1])
