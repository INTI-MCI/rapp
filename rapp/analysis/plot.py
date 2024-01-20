import os

import numpy as np

import matplotlib.ticker as tck
from matplotlib import pyplot as plt

from rapp.utils import create_folder

plt.style.use('style.mplstyle')

FOLDER = 'output-plots'


class Plot:
    """Encapsulates the creation of plots."""
    def __init__(self, title='', ylabel=None, xlabel=None, ysci=False, xint=False, folder=FOLDER):
        self._fig, self._ax = plt.subplots(figsize=(4, 4))
        self._ax.set_title(title, size=12)

        if ysci:
            self._ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

        if xint:
            self._ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        if ylabel is not None:
            self._ax.set_ylabel(ylabel)
            self._ax.set_xlabel(xlabel)

        self._folder = folder

    def add_data(self, xs, ys=None, style='o', ms=5, mew=0.5, mfc='None', xrad=False, **kwargs):
        """Adds data to the plot."""

        ax = self._ax

        if ys is None:
            ys = xs
            xs = np.arange(1, ys.size + 1, step=1)

        if xrad:
            xs = xs / np.pi
            ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\\pi$'))
            ax.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))

        return ax.errorbar(xs, ys, fmt=style, ms=ms, mfc=mfc, mew=mew, **kwargs)

    def save(self, filename):
        """Saves the plot."""
        create_folder(self._folder)
        self._fig.savefig(os.path.join(self._folder, filename))

    def legend(self, fontsize=11, frameon=False, **kwargs):
        self._ax.legend(fontsize=fontsize, frameon=frameon, **kwargs)

    def show(self):
        """Shows the plot."""
        plt.show()

    def set_title(self, title):
        self._ax.set_title(title, size=12)

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
