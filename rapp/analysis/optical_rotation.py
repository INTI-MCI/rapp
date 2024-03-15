import re
import os
import logging

import numpy as np
from uncertainties import ufloat

from rapp.measurement import Measurement, REGEX_NUMBER_AFTER_WORD
from rapp.utils import round_to_n


logger = logging.getLogger(__name__)


def optical_rotation(folder1, folder2, method='NLS', avg_or=False, hwp=False):
    print("")
    logger.debug("Folder without optical active sample measurements {}...".format(folder1))
    logger.debug("Folder with optical active sample measurements {}...".format(folder2))

    initial_poisition = float(re.findall(REGEX_NUMBER_AFTER_WORD.format(word="hwp"), folder1)[0])
    final_position = float(re.findall(REGEX_NUMBER_AFTER_WORD.format(word="hwp"), folder2)[0])

    logger.debug("Initial position: {}°".format(initial_poisition))
    logger.debug("Final position: {}°".format(final_position))

    expected_or = (final_position - initial_poisition) * (-1)

    if hwp:
        expected_or = expected_or * 2

    logger.info("Expected optical rotation: {}°".format(expected_or))

    files_i = sorted([os.path.join(folder1, x) for x in os.listdir(folder1)])
    files_f = sorted([os.path.join(folder2, x) for x in os.listdir(folder2)])

    phase_diff_i = []
    phase_diff_f = []

    logger.info("Computing phase differences...")
    for k in range(len(files_f)):
        measurement_i = Measurement.from_file(files_i[k])
        measurement_f = Measurement.from_file(files_f[k])
        *head, res_i = measurement_i.phase_diff(method=method, fix_range=not hwp)
        *head, res_f = measurement_f.phase_diff(method=method, fix_range=not hwp)

        phase_diff_i.append(ufloat(res_i.value, res_i.u))
        phase_diff_f.append(ufloat(res_f.value, res_f.u))

    ors = []
    if avg_or:
        logger.info("Computing N optical rotations and taking average...")

        for k in range(len(files_f)):
            rotation = phase_diff_f[k] - phase_diff_i[k]
            logger.info("Optical rotation {}: {}°".format(k + 1, rotation))
            ors.append(rotation)

    else:
        logger.info("Averaging N phase differences and computing one optical rotation...")
        ors.append(np.mean(phase_diff_f) - np.mean(phase_diff_i))

    result = np.mean(ors)

    logger.info("Optical rotation measured: {}".format(result))
    logger.info("Error: {}".format(round_to_n(abs(expected_or) - abs(result.n), 4)))

    return ors


def main():
    hwp_datasets = [
        ('data/2023-12-22/hwp0/', 'data/2023-12-22/hwp4.5/'),
        ('data/2023-12-28/hwp0/', 'data/2023-12-28/hwp4.5/'),
        ('data/2023-12-29/hwp0/', 'data/2023-12-29/hwp4.5/'),
        ('data/2023-12-28/hwp0/', 'data/2023-12-28/hwp29/'),
        ('data/2024-03-05-repeatability/hwp0', 'data/2024-03-05-repeatability/hwp9')

    ]

    all_rotations = []
    for dataset in hwp_datasets:
        rotations = optical_rotation(*dataset, avg_or=True, hwp=True)
        all_rotations.extend(rotations)

    print("")
    all_rotations_u = [o.s for o in all_rotations]
    all_rotations_values = [o.n for o in all_rotations]

    # We assign as the OR measurement uncertainty, the maximum uncertainty obtained.
    measurement_u = max(all_rotations_u)
    logger.info("Measurement Uncertainty: {}°".format(measurement_u))

    logger.info("Repeatability for 9 degrees of optical rotation:")
    rotation_values = all_rotations_values[0:3]
    logger.debug("Values taken into account for repeatability: {}".format(rotation_values))
    repeatability_u = np.std(np.abs(rotation_values)) / np.sqrt(len(rotation_values))
    logger.info("Repeatability Uncertainty: {}°". format(repeatability_u))

    combined_u = np.sqrt(measurement_u ** 2 + repeatability_u ** 2)
    logger.info("Combined Uncertainty (k=2): {}°". format(combined_u * 2))
