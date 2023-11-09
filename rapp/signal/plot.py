import os

import numpy as np

import matplotlib.ticker as tck
from matplotlib import pyplot as plt

from rapp.utils import create_folder

#plt.style.use('style.mplstyle')


class Plot:
    """Encapsulates the creation of plots."""
    def __init__(self, title='', ylabel=None, xlabel=None, ysci=False, folder='output-plots'):
        self._fig, self._ax = plt.subplots()
        self._ax.set_title(title, size=12)

        if ysci:
            self._ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

        if ylabel is not None:
            self._ax.set_ylabel(ylabel)
            self._ax.set_xlabel(xlabel)

        self._folder = folder

    def add_data(self, xs, ys, style='o', mew=0.5, xrad=False, **kwargs):
        """Adds data to the plot."""

        ax = self._ax

        if xs is None:
            xs = np.arange(1, ys.size + 1, step=1)

        if xrad:
            xs = xs / np.pi
            ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\\pi$'))
            ax.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))

        ax.errorbar(xs, ys, fmt=style, mfc='None', mew=mew, **kwargs)

    def save(self, filename):
        """Saves the plot."""
        create_folder(self._folder)
        self._fig.savefig(os.path.join(self._folder, filename))

    def legend(self, fontsize=10, **kwargs):
        self._ax.legend(fontsize=fontsize, **kwargs)

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
