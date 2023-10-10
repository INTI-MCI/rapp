import os

import numpy as np

import matplotlib.ticker as tck
from matplotlib import pyplot as plt

from rapp.utils import create_folder

plt.style.use('style.mplstyle')


class Plot:
    """Encapsulates the creation of plots."""

    def __init__(self, title='', ylabel=None, xlabel=None, folder='output'):
        self._fig, self._ax = plt.subplots()
        self._ax.set_title(title)

        if ylabel is not None:
            self._ax.set_ylabel(ylabel)
            self._ax.set_xlabel(xlabel)

        self._folder = folder

    def add_data(self, xs, ys, style='o', color='k', mew=0.8, lw=0.8, label=None, xrad=False):
        """Adds data to the plot."""

        ax = self._ax

        if xrad:
            xs = xs / np.pi
            ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\\pi$'))
            ax.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))

        ax.plot(xs, ys, style, ms=5, mfc='None', mew=mew, lw=lw, color=color, label=label)

    def save(self, filename):
        """Saves the plot."""
        create_folder(self._folder, overwrite=False)
        self._fig.savefig(os.path.join(self._folder, filename))

    def legend(self):
        self._ax.legend(fontsize=10)

    def show(self):
        """Shows the plot."""
        plt.show()

    def clear(self):
        """Clears the plot."""
        plt.close()
        title = self._ax.get_title()
        xlabel = self._ax.get_xlabel()
        ylabel = self._ax.get_ylabel()

        self._fig, self._ax = plt.subplots()
        self._ax.set_title(title)
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)

    def close(self):
        plt.close()
