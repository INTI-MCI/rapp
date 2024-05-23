import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import date

import numpy as np

from rich.progress import track

from rapp import constants as ct
from rapp.adc import ADC
from rapp.data_file import DataFile
from rapp.mocks import PM100Mock
from rapp.motion_controller import ESP301
from rapp.rotary_stage import RotaryStage, RotaryStageError
from rapp.utils import split_number_to_list
from rapp.pm100 import PM100


# Always truncate arrays when printing, without scientific notation.
np.set_printoptions(threshold=0, edgeitems=5, suppress=True)
logger = logging.getLogger(__name__)


ADC_WIN_PORT = 'COM3'
ADC_LINUX_PORT = '/dev/ttyACM0'

ADC_BAUDRATE = 57600
ADC_TIMEOUT = 0.1
ADC_WAIT = 2
ADC_SAMPLE_RATE = 840

THORLABS_PM100_VISA = "USB0::4883::32889::P1000529::0::INSTR"
THORLABS_PM100_TIME_PER_SAMPLE_MS = 3

MOTION_CONTROLLER_PORT = "COM4"
# MOTION_CONTROLLER_PORT = '/dev/ttyACM0'
MOTION_CONTROLLER_BAUDRATE = 921600

LOG_FILENAME = "rapp.log"


def resolve_adc_port():
    if sys.platform == 'linux':
        return ADC_LINUX_PORT

    return ADC_WIN_PORT


FILE_DELIMITER = ","
FILE_COLUMNS = ["ANGLE", "CH0", "CH1"]
FILE_NORMALIZATION_COLUMN_NAME = "NORM"
FILE_HEADER = (
    "#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#\n"
    "#~~~~~~~~~~~~~~ RAPP measurements | INTI ~~~~~~~~~~~~~~#\n"
    "#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#"
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
        adc: ADC, analyzer: RotaryStage, hwp: RotaryStage, data_file: DataFile,
        normalization_detector: PM100 = None, wait: int = 10
    ):
        self._adc = adc
        self._analyzer = analyzer
        self._hwp = hwp
        self._data_file = data_file
        self._norm_det = normalization_detector
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

        analyzer_bar = True
        self._adc.progressbar = False  # We disable lower level progress bar.
        if len(self._analyzer) == 1:
            self._adc.progressbar = True
            analyzer_bar = False

        failures = 0

        self._hwp.reset()
        for hwp_position in self._hwp:
            rep = 1
            while rep < reps + 1:
                logger.info("HWP angle: {}°, repetition {}/{}".format(hwp_position, rep, reps))

                self._analyzer.reset()
                self._data_file.open(self._build_data_filename(rep, hwp_position))

                p_desc = 'rep no. {}: '.format(rep)
                try:
                    for position in track(
                        self._analyzer, style='white', description=p_desc, disable=not analyzer_bar
                    ):
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

        self.close()

        return failures

    def read_samples(self, samples: int, chunk_size=0):
        """Reads samples in chunks at current position.

        Args:
            samples: number of samples to read.
            chunk_size: reads data in chunks of this size. If 0, no chunks are used.
        """
        n_samples = [samples]
        if chunk_size > 0:
            n_samples = split_number_to_list(num=samples, size=chunk_size)

        for samples in n_samples:
            if self._norm_det is not None:
                self._norm_det.set_average_count(self._amount_samples_pm100(samples))
                self._norm_det.start_measurement()

            acquired_samples = self._adc.acquire(samples, flush=True)
            if self._norm_det is not None:
                normalization_value = self._norm_det.fetch_measurement()
                acquired_samples = [(*acq_s, normalization_value) for acq_s in acquired_samples]
            yield acquired_samples

    def close(self):
        self._adc.close()
        self._analyzer.close()
        self._norm_det.close()

    def _build_data_filename(self, rep, hwp_position=None):
        filename = "rep{}.csv".format(rep)

        if hwp_position is not None:
            hwp_prefix = "hwp{}".format(hwp_position)
            filename = "{}-{}".format(hwp_prefix, filename)

        return filename

    def _add_data_to_file(self, data, position=None):
        logger.debug("Writing data to file...")
        for row in data:
            self._data_file.add_row([position] + list(row))

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

    def _amount_samples_pm100(self, samples):
        n_channels = int(self._adc._ch0) + int(self._adc._ch1)
        delay_adc = samples * n_channels / ADC_SAMPLE_RATE
        return int(delay_adc / THORLABS_PM100_TIME_PER_SAMPLE_MS * 1e3)


def run(
    samples: int = 169,
    cycles: float = 0,
    step: float = 45,
    reps: int = 1,
    delay_position: float = 0,
    velocity: float = 4,
    chunk_size: int = 500,
    no_ch0: bool = False,
    no_ch1: bool = False,
    prefix: str = 'test',
    mock_esp: bool = False,
    mock_adc: bool = False,
    mock_pm100: bool = False,
    overwrite: bool = False,
    hwp_cycles: float = 0,
    hwp_step: float = 45,
    hwp_delay_position: float = 5,
    mc_wait: float = 15,
    work_dir: str = ct.WORK_DIR
):

    metadata = locals().copy()
    del metadata['work_dir']

    params = "cycles{}-step{}-samples{}".format(cycles, step, samples)
    measurement_name = f"{date.today()}-{prefix}-{params}"
    measurement_dir = Path(work_dir).joinpath(ct.OUTPUT_FOLDER_DATA, measurement_name)
    os.makedirs(measurement_dir, exist_ok=True)

    logger.info("Writing metadata...")
    metadata_file = measurement_dir.joinpath("metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    logger.info("Connecting to ESP Motion Controller...")
    motion_controller = ESP301.build(
        MOTION_CONTROLLER_PORT, b=MOTION_CONTROLLER_BAUDRATE,
        useaxes=[1, 2], mock_serial=mock_esp)

    logger.info("Connecting to ADC...")
    adc = ADC.build(
        resolve_adc_port(), b=ADC_BAUDRATE, timeout=ADC_TIMEOUT, wait=ADC_WAIT,
        ch0=not no_ch0,
        ch1=not no_ch1,
        mock_serial=mock_adc)

    # Search for Normalization Detector and build
    if mock_pm100:
        logger.warning("Using PM100 mock object.")
        pm100 = PM100Mock(THORLABS_PM100_VISA)
    else:
        logger.info("Connecting to Thorlabs PM100.")
        pm100 = PM100.build(THORLABS_PM100_VISA)
        if pm100 is None:
            logger.info("PM100 not detected.")

    logger.info("Connecting Analyzer...")
    analyzer = RotaryStage(
        motion_controller, cycles, step, delay_position, velocity, axis=1, name='Analyzer')

    logger.info("Connecting HalfWavePlate...")
    hwp = RotaryStage(
        motion_controller, hwp_cycles, hwp_step, hwp_delay_position, axis=2, name='HalfWavePlate')

    logger.info("Building DataFile...")
    normalization_column = [] if pm100 is None else FILE_NORMALIZATION_COLUMN_NAME
    file_columns = FILE_COLUMNS + normalization_column
    data_file = DataFile(
        overwrite, header=FILE_HEADER, column_names=file_columns, delimiter=FILE_DELIMITER,
        output_dir=measurement_dir)

    logger.info("Building Polarimeter...")
    polarimeter = Polarimeter(
        adc, analyzer, hwp, data_file, normalization_detector=pm100, wait=mc_wait
    )

    logger.info("Starting measurement...")
    polarimeter.start(samples, chunk_size=chunk_size, reps=reps)
