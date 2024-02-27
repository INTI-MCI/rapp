import os
import sys
import math
import logging

from datetime import date, datetime

import numpy as np

from rapp import constants as ct
from rapp.motion_controller import ESP301
from rapp.adc import ADC
from rapp.rotary_stage import RotaryStage

from rapp.data_file import DataFile
from rapp.mocks import SerialMock
from rapp.utils import progressbar

# Always truncate arrays when printing, without scientific notation.
np.set_printoptions(threshold=0, edgeitems=5, suppress=True)
logger = logging.getLogger(__name__)


ADC_WIN_DEVICE = 'COM5'
ADC_LINUX_DEVICE = '/dev/ttyACM0'

ADC_BAUDRATE = 57600
ADC_TIMEOUT = 0.1
ADC_WAIT = 2

MOTION_CONTROLLER_PORT = "COM4"
MOTION_CONTROLLER_BAUDRATE = 921600


def resolve_adc_port():
    if sys.platform == 'linux':
        return ADC_LINUX_DEVICE

    return ADC_WIN_DEVICE


FILE_DELIMITER = "\t"
FILE_COLUMN_NAMES = ["ANGLE", "CH0", "CH1", "DATETIME"]
FILE_NAME = "cycles{}-step{}-samples{}-rep{}.txt"
FILE_HEADER = (
    "#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#\n"
    "#~~~~~~~~~ RAPP measurements | INTI {date} ~~~~~~~~#\n"
    "#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#\n"
    "# HWP     : {}\n"
    "# cycles  : {}\n"
    "# step    : {}\n"
    "# samples : {}\n"
    "# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#"
)


class Polarimeter:
    """High accuracy polarimeter for optical rotation measurements.

    Args:
        adc: acquisition device.
        analyzer: rotating analyzer.
        hwp: rotating half wave plate.
        data_file: handles whe file writing.
    """
    def __init__(self, adc: ADC, analyzer: RotaryStage, hwp: RotaryStage, data_file: DataFile):
        self._adc = adc
        self._analyzer = analyzer
        self._hwp = hwp
        self._data_file = data_file

    def start(self, samples, chunk_size: int = 0, reps: int = 1):
        """Collects measurements rotating the analyzer and saves them in a data file.

        Args:
            samples: number of samples per analyzer position.
            chunk_size: measure data in chunks of this size. If 0, no chunks are used.
            reps: number of repetitions.
        """

        logger.info("Samples to measure in each analyzer position: {}.".format(samples))
        logger.info("Maximum chunk size: {}.".format(chunk_size))

        self._hwp.set_home(0)

        for hwp_position in self._hwp:
            for rep in range(1, reps + 1):
                logger.info("HWP angle: {}°, repetition {}/{}".format(hwp_position, rep, reps))

                self._setup_data_file(samples, rep, hwp_position)
                self._adc.progressbar = False  # We disable lower level progress bar.
                progressbar_desc = '{} cycle(s): '.format(self._analyzer.cycles)

                for position in progressbar(self._analyzer, desc=progressbar_desc):
                    for data_chunk in self.read_samples(samples, chunk_size):
                        self._add_data_to_file(data_chunk, position=position)

                self._data_file.close()
                self._analyzer.reset()

                logger.info("Done!")
                logger.info("Results in file: {}".format(self._data_file.path))

        return self._data_file.path

    def read_samples(self, samples: int, chunk_size=0):
        """Reads samples in chunks at current position.

        Args:
            samples: number of samples to read.
            chunk_size: reads data in chunks of this size. If 0, no chunks are used.
        """
        if chunk_size > 0:
            n_samples = [samples] * math.ceil(samples / chunk_size)

        for samples in n_samples:
            yield self._adc.acquire(samples, flush=True)

    def close(self):
        self._adc.close()
        self._analyzer.close()

    def _build_data_file_name(self, samples, rep, hwp_position=None):
        filename = FILE_NAME.format(self._analyzer.cycles, self._analyzer.step, samples, rep)

        if hwp_position is not None:
            hwp_prefix = "hwp{}".format(hwp_position)
            filename = "{}-{}".format(hwp_prefix, filename)

        return filename

    def _setup_data_file(self, samples, rep, hwp_position):
        self._data_file.header = self._build_data_file_header(samples, hwp_position)
        self._data_file.column_names = FILE_COLUMN_NAMES
        self._data_file.open(self._build_data_file_name(samples, rep, hwp_position))

    def _build_data_file_header(self, samples, hwp_position):
        return FILE_HEADER.format(
            hwp_position,
            self._analyzer.cycles,
            self._analyzer.step,
            samples,
            date=date.today()
        )

    def _add_data_to_file(self, data, position=None):
        logger.info("Writing data to file...")
        time = datetime.now().isoformat()  # TODO: get real measurement time from ADC?
        for row in data:
            self._data_file.add_row([position] + list(row) + [time])


def main(
    samples=10, cycles=0, step=45, delay_position=1, velocity=2, no_ch0=False, no_ch1=False,
    chunk_size=2000, prefix='test', mock_esp=False, mock_adc=False, plot=False, overwrite=False,
    hwp_cycles=0, hwp_step=45, hwp_delay=5, reps=1, work_dir=ct.WORK_DIR
):

    # Build Motion Controller
    if mock_esp:
        logger.warning("Using ESP Motion Controller mock object.")
        motion_controller = ESP301(SerialMock())
    else:
        logger.info("Connecting to ESP Motion Controller...")
        motion_controller = ESP301.build(
            MOTION_CONTROLLER_PORT, b=MOTION_CONTROLLER_BAUDRATE, useaxes=[1, 2], reset=True
        )

    # Build ADC
    if mock_adc:
        logger.warning("Using ADC mock object.")
        adc = ADC(SerialMock(), ch0=not no_ch0, ch1=not no_ch1)
    else:
        logger.info("Connecting to ADC...")
        adc = ADC.build(
            resolve_adc_port(),
            b=ADC_BAUDRATE, timeout=ADC_TIMEOUT, wait=ADC_WAIT, ch0=not no_ch0, ch1=not no_ch1
        )

    # Build Analyzer
    analyzer = RotaryStage(
        motion_controller, cycles, step, delay_position, velocity, axis=1
    )

    # Build HalfWavePlate
    hwp = RotaryStage(motion_controller, hwp_cycles, hwp_step, delay_position=hwp_delay, axis=2)

    # Build DataFile
    output_dir = os.path.join(work_dir, ct.OUTPUT_FOLDER_DATA)
    data_file = DataFile(overwrite, prefix=prefix, delimiter=FILE_DELIMITER, output_dir=output_dir)

    # Build Polarimeter
    polarimeter = Polarimeter(adc, analyzer, hwp, data_file)

    # Start polarimeter measurement
    polarimeter.start(samples, chunk_size=chunk_size, reps=reps)
    polarimeter.close()
