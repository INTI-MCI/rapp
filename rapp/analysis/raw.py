import os
import logging
import numpy as np

from rapp import constants as ct
from rapp.utils import create_folder
from rapp.measurement import Measurement
from rapp.analysis.plot import Plot
from rapp.analysis.noise import plot_histogram_and_pdf


logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def plot_raw_from_file(filepath, work_dir=ct.WORK_DIR, **kwargs):
    measurement = Measurement.from_file(filepath)

    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    output_filename = os.path.basename(filepath)[:-4]
    create_folder(output_folder)

    return plot_raw(measurement, output_folder, output_filename, **kwargs)


def plot_raw(
    measurement, output_folder, output_filename, no_ch0=False, no_ch1=False, show=False
):

    s1 = np.array(measurement.ch0())[:150000]
    s2 = np.array(measurement.ch1())

    logger.info("STD: {}".format(np.std(s1)))

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_ANGLE, folder=output_folder)

    plot.set_title(measurement.parameters_string())

    if not no_ch0:
        # plot.add_data(s1, style='-', color='k', lw=1.5, label='CH0')
        plot._ax.hist(s1, bins=4)
    if not no_ch1:
        plot.add_data(s2, style='--', color='k', lw=1.5, label='CH1')

    # plot._ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    plot.legend(loc='upper right', fontsize=12, frameon=True)

    plot.save(filename="{}.png".format(output_filename))

    if show:
        plot.show()

    plot.close()
