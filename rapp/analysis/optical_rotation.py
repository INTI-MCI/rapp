import re
import os
import logging

import numpy as np
from uncertainties import ufloat

from rapp.analysis.parser import REGEX_NUMBER_AFTER_WORD
from rapp.measurement import Measurement

logger = logging.getLogger(__name__)


def optical_rotation(folder_i, folder_f, method='ODR', hwp=False):
    print("")
    initial_poisition = float(re.findall(REGEX_NUMBER_AFTER_WORD.format(word="hwp"), folder_i)[0])
    final_position = float(re.findall(REGEX_NUMBER_AFTER_WORD.format(word="hwp"), folder_f)[0])

    logger.debug("Initial position: {}°".format(initial_poisition))
    logger.debug("Final position: {}°".format(final_position))

    or_angle = (final_position - initial_poisition)
    logger.info("Expected optical rotation: {}°".format(or_angle))

    logger.debug("Folder without optical active sample measurements {}...".format(folder_i))
    logger.debug("Folder with optical active sample measurements {}...".format(folder_f))

    ors = []
    files_i = sorted([os.path.join(folder_i, x) for x in os.listdir(folder_i)])
    files_f = sorted([os.path.join(folder_f, x) for x in os.listdir(folder_f)])

    for k in range(len(files_i)):
        measurement_i = Measurement.from_file(files_i[k])
        measurement_f = Measurement.from_file(files_f[k])
        *head, res_i = measurement_i.phase_diff(method=method, fix_range=not hwp)
        *head, res_f = measurement_f.phase_diff(method=method, fix_range=not hwp)

        res_i = ufloat(res_i.value, res_i.u)
        res_f = ufloat(res_f.value, res_f.u)

        optical_rotation = res_f - res_i

        if hwp:
            optical_rotation = ufloat(optical_rotation.n * 0.5, optical_rotation.s)

        logger.info("Optical rotation {}: {}°".format(k + 1, optical_rotation))

        ors.append(optical_rotation)

    N = len(ors)
    avg_or = sum(ors) / N

    values = [o.n for o in ors]
    repeatability_u = np.std(values) / np.sqrt(len(values))

    logger.info("Optical rotation measured (average): {}".format(avg_or))
    logger.debug("Repeatability uncertainty: {}".format(repeatability_u))
    logger.info("Error: {}".format(abs(or_angle) - abs(avg_or)))

    return avg_or, ors


def main():
    _, ors1 = optical_rotation('data/22-12-2023/hwp0/', 'data/22-12-2023/hwp4.5/', hwp=True)
    _, ors2 = optical_rotation('data/28-12-2023/hwp0/', 'data/28-12-2023/hwp4.5/', hwp=True)
    _, ors3 = optical_rotation('data/28-12-2023/hwp0/', 'data/28-12-2023/hwp29/', hwp=True)
    _, ors4 = optical_rotation('data/29-12-2023/hwp0/', 'data/29-12-2023/hwp-9/')

    print("")
    measurement_u = max([o.s for o in ors1 + ors2 + ors3 + ors4])
    logger.info("Measurement Uncertainty: {}°".format(measurement_u))

    all_45_ors = [o.n for o in ors1 + ors2]
    logger.debug("Values taken into account for repeatability: {}".format(all_45_ors))

    repeatability_u = np.std(np.abs(all_45_ors)) / len(all_45_ors)
    logger.info("Repeatability Uncertainty: {}°". format(repeatability_u))

    combined_u = np.sqrt(measurement_u ** 2 + repeatability_u ** 2)
    logger.info("Combined Uncertainty (k=2): {}°". format(combined_u * 2))
