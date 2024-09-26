import os
import logging
import numpy as np

import matplotlib.ticker as tck
from matplotlib import pyplot as plt

from rapp.utils import create_folder

logger = logging.getLogger(__name__)

plt.style.use("style.mplstyle")

FOLDER = "output-plots"


class Plot:
    """Encapsulates the creation of plots."""

    def __init__(
        self, nrows=1, ncols=1, title="", ylabel=None, xlabel=None, ysci=False, yoom=0, xint=False,
        folder=FOLDER
    ):
        self.current_subplot_flat = 0
        self._nrows = nrows
        self._ncols = ncols
        self._fig, self._axs = plt.subplots(nrows, ncols, figsize=(4, 4), squeeze=False)
        self.all_axs(lambda ax, t: ax.set_title(t, size=12), title)

        if ysci:
            self.all_axs(
                lambda ax: ax.ticklabel_format(style="sci", scilimits=(yoom, yoom), axis="y"))

        if xint:
            self.all_axs(
                lambda ax: ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True)))

        if ylabel is not None:
            self.all_axs(lambda ax, yl: ax.set_ylabel(yl), ylabel)
            self.all_axs(lambda ax, xl: ax.set_xlabel(xl), xlabel)

        self._folder = folder

    def all_axs(self, func_ax_data, data=None):
        data = self.vet_var_for_axs(data)
        output = np.full((self._nrows, self._ncols), None)

        for kr in range(self._nrows):
            for kc in range(self._ncols):
                if data is None:
                    output[kr, kc] = func_ax_data(self._axs[kr, kc])
                else:
                    output[kr, kc] = func_ax_data(self._axs[kr, kc], data[kr, kc])
        return output

    @property
    def the_ax(self):
        return self._axs[0, 0]

    def vet_var_for_axs(self, var):
        if var is None:
            return var
        if isinstance(var, list):
            if len(var) == self._nrows and all(len(row) == self._ncols for row in var):
                return var
        elif isinstance(var, np.ndarray):
            if var.shape == (self._nrows, self._ncols):
                return var
        elif isinstance(var, (str, int, float)):
            return np.full((self._nrows, self._ncols), var)
        raise ValueError("Not valid input or wrong dimensions (do not match subplot dimensions).")

    def add_data(self, xs, ys=None, style="o", ms=5, mew=0.5, xrad=False, nrow=0, ncol=0,
                 **kwargs):
        """Adds data to the plot."""

        ax = self._axs[nrow, ncol]

        if ys is None:
            ys = xs
            xs = np.arange(1, len(ys) + 1, step=1)

        if xrad:
            xs = xs / np.pi
            ax.xaxis.set_major_formatter(tck.FormatStrFormatter("%g $\\pi$"))
            ax.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))

        return ax.errorbar(xs, ys, fmt=style, ms=ms, mew=mew, **kwargs)

    def add_image(self, xys, im=None, subplot_rc=None, **kwargs):
        if im is None:
            im = xys
            extent = None
        else:
            half_step_rows = (xys[1][1] - xys[1][0]) / 2
            half_step_cols = (xys[0][1] - xys[0][0]) / 2
            extent = (xys[0][0] - half_step_cols,
                      xys[0][-1] + half_step_cols,
                      xys[1][-1] + half_step_rows,
                      xys[1][0] - half_step_rows)
        if subplot_rc is None:
            subplot_rc = np.unravel_index(self.current_subplot_flat, (self._nrows, self._ncols))
        else:
            self.current_subplot_flat = np.ravel_multi_index(subplot_rc,
                                                             (self._nrows, self._ncols))

        self.current_subplot_flat += 1
        self.current_subplot_flat = self.current_subplot_flat % (self._nrows * self._ncols)

        return self._axs[subplot_rc].imshow(im, extent=extent, aspect='auto', **kwargs)

    def save(self, filename):
        """Saves the plot."""
        create_folder(self._folder)
        self._fig.savefig(os.path.join(self._folder, filename))

    def legend(self, fontsize=11, frameon=False, nrow=0, ncol=0, **kwargs):
        self._axs[nrow, ncol].legend(fontsize=fontsize, frameon=frameon, **kwargs)

    def show(self):
        """Shows the plot."""
        plt.show()

    def set_title(self, title, nrow=0, ncol=0):
        self._axs[nrow, ncol].set_title(title, size=12)

    def yaxis_set_major_formatter(self, yfmt):
        self.all_axs(lambda ax: ax.yaxis.set_major_formatter(yfmt))
        self.all_axs(lambda ax: ax.yaxis.set_major_locator(plt.MaxNLocator(2)))

    def set_formatter(self, fmt):
        self.all_axs(lambda ax: ax.xaxis.set_major_formatter(fmt))
        self.all_axs(lambda ax: ax.yaxis.set_major_formatter(fmt))

    def clear(self):
        """Clears the plot."""
        plt.close()
        titles = self.all_axs(lambda ax: ax.get_title())
        xlabels = self.all_axs(lambda ax: ax.get_xlabel())
        ylabels = self.all_axs(lambda ax: ax.get_ylabel())

        self._fig, self._axs = plt.subplots(self._nrows, self._ncols, figsize=(4, 4),
                                            squeeze=False)
        self.all_axs(lambda ax, t: ax.set_title(t), titles)
        self.all_axs(lambda ax, xl: ax.set_xlabel(xl), xlabels)
        self.all_axs(lambda ax, yl: ax.set_ylabel(yl), ylabels)

    def close(self):
        plt.close()

    def move(self, dxy_in_cm):
        try:
            tk_window = self._fig.canvas.manager.window
            x, y = tk_window.winfo_x(), tk_window.winfo_y()
            dpi = self._fig.dpi
            tk_window.geometry(f"+{x + int(dxy_in_cm[0] / 2.54 * dpi)}+"
                               f"{y + int(dxy_in_cm[1] / 2.54 * dpi)}")
        except AttributeError:
            logger.debug("Impossible to move image.", exc_info=True)
