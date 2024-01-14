import os

import numpy as np

from rapp import constants as ct
from rapp.utils import create_folder
from rapp.measurement import Measurement
from rapp.analysis.plot import Plot
from rapp.analysis.parser import parse_input_parameters_from_filepath

PARAMETER_STRING = "cycles={}, step={}°, samples={}."


def plot_raw(filepath, sep='\t', usecols=(0, 1, 2), no_ch0=False, no_ch1=False, show=False):
    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    measurement = Measurement.from_file(filepath)

    s1 = np.array(measurement.ch0())
    s2 = np.array(measurement.ch1())

    plot = Plot(ylabel=ct.LABEL_VOLTAGE, xlabel=ct.LABEL_ANGLE, folder=output_folder)

    plot.set_title(PARAMETER_STRING.format(*parse_input_parameters_from_filepath(filepath)))

    if not no_ch0:
        plot.add_data(s1, style='-', color='k', lw=1.5, label='CH0')

    if not no_ch1:
        plot.add_data(s2, style='--', color='k', lw=1.5, label='CH1')

    # plot._ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    plot.legend(loc='upper right', fontsize=12, frameon=True)

    plot.save(filename="{}.png".format(os.path.basename(filepath)[:-4]))

    if show:
        plot.show()

    plot.close()
