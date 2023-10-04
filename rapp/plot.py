import os

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

    def add_data(self, x_data, y_data, style='o', color='k', label=None):
        """Adds data to the plot."""

        ax = self._ax

        ax.plot(x_data, y_data, style, ms=5, mfc='None', color=color, label=label)
        ax.legend(loc='upper right', fontsize=10)

    def save(self, title=None):
        """Saves the plot. If title is provided, overrides the current plot title."""
        if title is not None:
            self._ax.set_title(title)

        create_folder(self._folder, overwrite=False)
        filename = f'{self._ax.get_title().lower().replace(" ", "_")}'
        self._fig.savefig(os.path.join(self._folder, filename))

    def show(self):
        """Shows the plot."""
        plt.show()

    def clear(self):
        """Clears the plot."""
        title = self._ax.get_title()
        xlabel = self._ax.get_xlabel()
        ylabel = self._ax.get_ylabel()

        self._fig, self._ax = plt.subplots()
        self._ax.set_title(title)
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)
