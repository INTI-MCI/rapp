import glob
import logging
from pathlib import Path

import numpy as np

from rapp.measurement import Measurement
from rapp.utils import round_to_n

logger = logging.getLogger(__name__)


def optical_rotation(folder1, folder2, method="DFT"):
    logger.debug("Folder without optical active sample measurements {}...".format(folder1))
    logger.debug("Folder with optical active sample measurements {}...".format(folder2))

    files_i = sorted(glob.glob(f"{folder1}/*.csv"))
    files_f = sorted(glob.glob(f"{folder2}/*.csv"))

    if not files_i:
        raise ValueError("Empty folder!: {}".format(folder1))

    if not files_f:
        raise ValueError("Empty folder!: {}".format(folder2))

    phase_diff_i = []
    phase_diff_f = []

    logger.debug("Computing phase differences...")
    for k in range(len(files_f)):
        logger.debug("Loading repetition {}".format(k + 1))
        measurement_i = Measurement.from_file(files_i[k])
        measurement_f = Measurement.from_file(files_f[k])
        *head, res_i = measurement_i.phase_diff(method=method, fix_range=True)
        *head, res_f = measurement_f.phase_diff(method=method, fix_range=True)

        phase_diff_i.append(res_i.value)
        phase_diff_f.append(res_f.value)

    optical_rotation = np.mean(phase_diff_f) - np.mean(phase_diff_i)

    return optical_rotation


def main():
    quartz_plate = {
        "nominal_value": 9.8,
        "measurements": [
            (
                "2024-07-04-tanda-1-no-quartz-vel-3-cycles1-step1-samples169/",
                "2024-07-04-tanda-1-quartz-vel-3-cycles1-step1-samples169/",
            ),
            (
                "2024-07-05-tanda-2-no-quartz-vel-3-cycles1-step1-samples169/",
                "2024-07-04-tanda-2-quartz-vel-3-cycles1-step1-samples169/",
            ),
        ],
    }

    nominal_value = quartz_plate["nominal_value"]
    logger.info("Nominal value: {}".format(nominal_value))

    for folder1, folder2 in quartz_plate["measurements"]:
        folder1 = Path("data").joinpath(folder1)
        folder2 = Path("data").joinpath(folder2)
        rapp = optical_rotation(folder1, folder2)
        error = round_to_n(abs(nominal_value - abs(rapp)), 4)
        logger.info("(RAPP, error): ({}, {})".format(rapp, error))
