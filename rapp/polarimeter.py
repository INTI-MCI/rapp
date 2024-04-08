import os
import sys
import time
import math
import logging
from pathlib import Path
from datetime import date, datetime

import numpy as np

from rapp import constants as ct
from rapp.adc import ADC
from rapp.data_file import DataFile
from rapp.mocks import SerialMock
from rapp.motion_controller import ESP301
from rapp.rotary_stage import RotaryStage, RotaryStageError
from rapp.utils import progressbar


# Always truncate arrays when printing, without scientific notation.
np.set_printoptions(threshold=0, edgeitems=5, suppress=True)
logger = logging.getLogger(__name__)


ADC_WIN_DEVICE = 'COM3'
ADC_LINUX_DEVICE = '/dev/ttyACM0'

ADC_BAUDRATE = 57600
ADC_TIMEOUT = 0.1
ADC_WAIT = 2

MOTION_CONTROLLER_PORT = "COM4"
# MOTION_CONTROLLER_PORT = '/dev/ttyACM0'
MOTION_CONTROLLER_BAUDRATE = 921600
MOTION_CONTROLLER_WAIT = 15  # Time to wait after error before reconnecting

LOG_FILENAME = "rapp-{datetime}.log"


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
        data_file: handles the file writing.
        wait: time to wait before reconnecting after motion controller error.
    """
    def __init__(
        self,
        adc: ADC, analyzer: RotaryStage, hwp: RotaryStage, data_file: DataFile, wait: int = 10
    ):
        self._adc = adc
        self._analyzer = analyzer
        self._hwp = hwp
        self._data_file = data_file
        self._wait = wait

    def start(self, samples, chunk_size: int = 0, reps: int = 1):
        """Collects measurements rotating the analyzer and saves them in a data file.

        Args:
            samples: number of samples per analyzer position.
            chunk_size: measure data in chunks of this size. If 0, no chunks are used.
            reps: number of repetitions.
        """

        logger.info("Samples to measure in each analyzer position: {}.".format(samples))
        logger.info("Maximum chunk size: {}.".format(chunk_size))

        failures = 0

        self._hwp.reset()
        for hwp_position in self._hwp:
            rep = 1
            while rep < reps + 1:
                logger.info("HWP angle: {}°, repetition {}/{}".format(hwp_position, rep, reps))

                self._analyzer.reset()
                self._setup_data_file(samples, rep, hwp_position)
                self._adc.progressbar = False  # We disable lower level progress bar.
                progressbar_desc = '{} cycle(s): '.format(self._analyzer.cycles)

                try:
                    for position in progressbar(self._analyzer, desc=progressbar_desc):
                        for data_chunk in self.read_samples(samples, chunk_size):
                            self._add_data_to_file(data_chunk, position=position)

                except RotaryStageError as e:
                    logger.warning("Motion Controller error: {}".format(e))
                    self._handle_motion_controller_error(hwp_position)
                    failures += 1
                    continue

                rep += 1
                self._data_file.close()

        logger.info("Done!")
        logger.info("Results in file: {}".format(self._data_file.path))
        logger.info("Total of {} failures with the Motion Controller.".format(failures))

        log_file = None
        handler = logging.getLogger().handlers[1]
        if hasattr(handler, 'baseFilename'):
            log_file = handler.baseFilename
            logger.info("The log is in: {}".format(log_file))

        self.close()

        return failures

    def read_samples(self, samples: int, chunk_size=0):
        """Reads samples in chunks at current position.

        Args:
            samples: number of samples to read.
            chunk_size: reads data in chunks of this size. If 0, no chunks are used.
        """
        n_samples = [samples]

        if chunk_size > 0 and samples > chunk_size:
            n_samples = [chunk_size] * math.ceil(samples / chunk_size)

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
        logger.debug("Writing data to file...")
        time = datetime.now().isoformat()  # TODO: get real measurement time from ADC?
        for row in data:
            self._data_file.add_row([position] + list(row) + [time])

    def _handle_motion_controller_error(self, hwp_position):
        logger.warning("Waiting {} seconds...".format(self._wait))
        time.sleep(self._wait)

        logger.warning("Reconnecting the Motion Controller...")
        # Accessing private attribute... bad design.
        # TODO: make polarimeter use only a MotionController object.
        self._analyzer._motion_controller.reconnect()

        logger.warning("Turning motors ON again...")
        self._analyzer.motor_on()
        self._hwp.motor_on()

        logger.warning("Setting HWP position to {}".format(hwp_position))
        self._hwp.set_home(hwp_position)

        logger.warning("Removing unfinished file...")
        self._data_file.remove()


def main(
    samples=169, cycles=0, step=45, delay_position=0, velocity=4, no_ch0=False, no_ch1=False,
    chunk_size=500, prefix='test', mock_esp=False, mock_adc=False, plot=False, overwrite=False,
    hwp_cycles=0, hwp_step=45, hwp_delay=5, reps=1,
    mc_wait=MOTION_CONTROLLER_WAIT, work_dir=ct.WORK_DIR
):

    log_filename = LOG_FILENAME.format(datetime=date.today().isoformat())
    handler = logging.FileHandler(os.path.join(work_dir, log_filename))
    logging.getLogger().addHandler(handler)

    logger.info("Connecting to ESP Motion Controller...")
    motion_controller = ESP301.build(
        MOTION_CONTROLLER_PORT, b=MOTION_CONTROLLER_BAUDRATE, useaxes=[1, 2], mock_serial=mock_esp
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
        motion_controller, cycles, step, delay_position, velocity, axis=1,
        name='Analyzer'
    )

    # Build HalfWavePlate
    hwp = RotaryStage(
        motion_controller, hwp_cycles, hwp_step, delay_position=hwp_delay, axis=2,
        name='HalfWavePlate')

    # Build DataFile
    output_dir = Path(str(work_dir)).joinpath(ct.OUTPUT_FOLDER_DATA)
    output_dir = str(output_dir)
    data_file = DataFile(overwrite, prefix=prefix, delimiter=FILE_DELIMITER, output_dir=output_dir)

    # Build Polarimeter
    polarimeter = Polarimeter(adc, analyzer, hwp, data_file, wait=mc_wait)

    # Start polarimeter measurement
    polarimeter.start(samples, chunk_size=chunk_size, reps=reps)
