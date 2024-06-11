import os
import logging
import numpy as np

from rapp import constants as ct
from rapp.utils import create_folder
from rapp.measurement import Measurement
from rapp.analysis.plot import Plot


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

    s1 = np.array(measurement.ch0())
    s2 = np.array(measurement.ch1())
    s3 = measurement.norm_data()

    # s1_n = s1/np.max(s1)
    # s3_n = s3/np.max(s3)
    if s3 is not None:
        logger.info(np.max(s3))

    logger.info("STD: {}".format(np.std(s1)))

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_N_SAMPLE, folder=output_folder)

    plot.set_title(measurement.parameters_string())

    if not no_ch0:
        plot.add_data(s1, style='-', color='k', lw=1.5, label='CH0')

    if not no_ch1:
        plot.add_data(s2, style='--', color='k', lw=1.5, label='CH1')

    if s3 is not None:
        s3_range = np.max(s3) - np.min(s3)
        s3_factor = np.max(s1) / s3_range
        plot.add_data((s3 - np.min(s3)) * s3_factor, style='.-', lw=1.5,
                      label=f'NORM (range:{s3_range:.2E}), avg:{s3.mean():.2E}')

    # plot._ax.hist(s1, bins=4)
    # plot._ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    plot.legend(loc='upper right', fontsize=12, frameon=True)

    plot.save(filename="{}.png".format(output_filename))

    if show:
        plot.show()

    plot.close()
