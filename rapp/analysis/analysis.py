import os
import logging


from rapp import constants as ct
from rapp.utils import create_folder

from rapp.analysis import noise, drift, optical_rotation

logger = logging.getLogger(__name__)


ANALYSIS_NAMES = [
    'all',
    'darkcurrent',
    'noise',
    'noise_cos',
    'drift',
    'OR',
]


def main(name, show=False):
    output_folder = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_PLOTS)
    create_folder(output_folder)

    if name not in ANALYSIS_NAMES:
        raise ValueError("Analysis with name {} not implemented".format(name))

    if name in ['all', 'darkcurrent']:
        noise.plot_noise_with_laser_off(output_folder, show=show)

    if name in ['all', 'noise']:
        noise.plot_noise_with_laser_on(output_folder, show=show)

    if name in ['all', 'noise_cos']:
        noise.plot_noise_with_signal(show=show)

    if name in ['drift']:
        drift.plot_drift(output_folder, show=show)

    if name == 'OR':
        optical_rotation.main()


if __name__ == '__main__':
    main()
