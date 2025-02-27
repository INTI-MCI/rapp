import os
import sys
import time
import datetime
import json
import logging
import warnings

from pathlib import Path
from datetime import date, timedelta

import numpy as np
import schedule
from rich.progress import track

from rapp import constants as ct
from rapp.adc import ADC
from rapp.data_file import DataFile
from rapp.motion_controller import ESP301
from rapp.rotary_stage import RotaryStage, RotaryStageError
from rapp.utils import split_number_to_list, timing
from rapp.pm100 import PM100, PM100Error


# Always truncate arrays when printing, without scientific notation.
np.set_printoptions(threshold=0, edgeitems=5, suppress=True)
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module='pyvisa_py'
)


ADC_PORT_WIN = 'COM3'
ADC_PORT_LINUX = '/dev/ttyACM0'

ADC_BAUDRATE = 57600
ADC_TIMEOUT = 2
ADC_TIMEOUT_OPEN = 5

THORLABS_PM100_VISA_LINUX = "USB0::4883::32889::P1000529::0::INSTR"
THORLABS_PM100_VISA_WIN = 'USB0::0x1313::0x8079::P1000529::INSTR'

MOTION_CONTROLLER_PORT_WIN = "COM4"
MOTION_CONTROLLER_PORT_LINUX = '/dev/ttyACM0'
MOTION_CONTROLLER_BAUDRATE = 921600

LOG_FILE_NAME = "rapp.log"
LOG_FILE_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def resolve_adc_port():
    if sys.platform == 'linux':
        return ADC_PORT_LINUX

    return ADC_PORT_WIN


def resolve_pm100_resource():
    if sys.platform == 'linux':
        return THORLABS_PM100_VISA_LINUX

    return THORLABS_PM100_VISA_WIN


FILE_DELIMITER = ","
FILE_COLUMNS = ["ANGLE", "CH0", "CH1", "NORM"]
FILE_HEADER = (
    "#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#\n"
    "#~~~~~~~~~~~~~~ RAPP measurements | INTI ~~~~~~~~~~~~~~#\n"
    "#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#"
)

TEMP_COLUMNS = ["ANGLE", "TEMP", "HWP-POS", "REP"]
TEMP_HEADER = "Tiempo-espera-{} s"

TEMP_CORRECTION_FILE = "workdir/output-data/2024-11-14-temperature-correction-parameters.json"


class Polarimeter:
    """High accuracy polarimeter for optical rotation measurements.

    Args:
        adc: acquisition device.
        analyzer: rotating analyzer.
        hwp: rotating half wave plate.
        data_file: handles the file writing.
        temp_file: handles the temperature file writing.
        temp_correction_file: temperature correction file.
        norm_det: normalization detector.
        wait: time to wait before reconnecting after motion controller error.
    """

    def __init__(
        self,
        adc: ADC, analyzer: RotaryStage, hwp: RotaryStage, data_file: DataFile,
        temp_file: DataFile = None, temp_correction_file: str = TEMP_CORRECTION_FILE,
        norm_det: PM100 = None, wait: int = 10
    ):
        self._adc = adc
        self._analyzer = analyzer
        self._hwp = hwp
        self._data_file = data_file
        self._temp_file = temp_file
        self._temp_correction_file = temp_correction_file
        self._norm_det = norm_det
        self._wait = wait

    def start(self, samples, chunk_size: int = 0, reps: int = 1, temp_wait: int = 60,
              temp_correction: str = 'bias'):
        """Collects measurements rotating the analyzer and saves them in a data file.

        Args:
            samples: number of samples per analyzer position.
            chunk_size: measure data in chunks of this size. If 0, no chunks are used.
            reps: number of repetitions.
            temp_wait: time to wait (in seconds) between room temperature measurements.
            temp_correction: temperature correction method.
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

        parameters = {
            "position": 0,
            "hwp_position": 0,
            "rep": 1,
            "temp_correction": temp_correction
        }

        parameters_req_temperature = {
            "position_r": 0,
            "hwp_position_r": 0,
            "rep_r": 1,
            "temp_correction_r": temp_correction
        }

        if temp_wait < 1:
            logger.warning("Temperature wait time is too short. It will be set to 1 seconds.")
            temp_wait = 1

        temperature_requested = [False]

        schedule_request = schedule.Scheduler()
        schedule_request.every(temp_wait).seconds.do(self.request_temperature,
                                                     parameters_req_temperature, parameters,
                                                     temperature_requested=temperature_requested)

        schedule_read = schedule.Scheduler()
        schedule_read.every(temp_wait).seconds.do(self.read_temperature,
                                                  parameters_req_temperature,
                                                  temperature_requested=temperature_requested,
                                                  write=True)

        self._temp_file.open("temperature.csv")
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
                        parameters["position"] = position
                        parameters["hwp_position"] = hwp_position
                        parameters["rep"] = rep

                        schedule_read.run_pending()

                        for data_chunk in self.read_samples(samples, chunk_size):
                            self._add_data_to_file(data_chunk, position=position)

                        schedule_request.run_pending()

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
                self._norm_det.start_measurement()

            acquired_samples = self._adc.acquire(samples, flush=True)
            logger.debug("Read samples at: {}".format(datetime.datetime.now()))
            if self._norm_det is not None:
                normalization_value = self._norm_det.fetch_measurement()
                acquired_samples = [(*acq_s, normalization_value) for acq_s in acquired_samples]

            yield acquired_samples

    def request_temperature(self, parameters_req_temperature={}, parameters={},
                            temperature_requested=[False]):
        if not temperature_requested[0]:
            temperature_requested[0] = self._adc.request_temperature()
            logger.debug("Request temperature at: {}".format(datetime.datetime.now()))

            parameters_req_temperature["position_r"] = parameters["position"]
            parameters_req_temperature["hwp_position_r"] = parameters["hwp_position"]
            parameters_req_temperature["rep_r"] = parameters["rep"]
            parameters_req_temperature["temp_correction_r"] = parameters["temp_correction"]

    def read_temperature(self, parameters_req_temperature={}, temperature_requested=[True],
                         write=True):
        if temperature_requested[0]:
            acquired_temperature, temperature_requested[0] = self._adc.read_temperature()
            logger.debug("Read temperature at: {}".format(datetime.datetime.now()))

            if parameters_req_temperature["temp_correction_r"] == 'bias':
                acquired_temperature = self.temperature_bias_correction(
                    filepath=TEMP_CORRECTION_FILE, temperature=acquired_temperature
                )
            elif parameters_req_temperature["temp_correction_r"] == 'linear':
                acquired_temperature = self.temperature_linear_correction(
                    filepath=TEMP_CORRECTION_FILE, temperature=acquired_temperature
                )

            data = ([parameters_req_temperature["position_r"]]
                    + [round(acquired_temperature, 4)]
                    + [parameters_req_temperature["hwp_position_r"]]
                    + [parameters_req_temperature["rep_r"]]
                    )

            if write:
                self._temp_file.add_row(data)
            logger.debug("Temperature: {}".format(acquired_temperature))

    def temperature_bias_correction(self, filepath=TEMP_CORRECTION_FILE, temperature=[]):
        with open(filepath, 'r') as f:
            json_data = json.load(f)

        bias = json_data['correction_parameters']['bias']
        temperature = temperature[0]
        return temperature - float(bias)

    def temperature_linear_correction(self, filepath=TEMP_CORRECTION_FILE, temperature=[]):
        with open(filepath, 'r') as f:
            json_data = json.load(f)

        slope = json_data['correction_parameters']['A']
        intercept = json_data['correction_parameters']['b']
        temperature = temperature[0]
        return temperature * float(slope) + float(intercept)

    def close(self):
        self._adc.close()
        self._analyzer.close()

        if self._norm_det is not None:
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


def setup_log_file(filepath):
    formatter = logging.Formatter(LOG_FILE_FORMAT)
    fh = logging.FileHandler(filepath, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def run(
    samples: int = 169,
    cycles: float = 0,
    step: float = 45,
    reps: int = 1,
    delay_position: float = 0,
    velocity: float = 4,
    acceleration: float = 8,
    deceleration: float = 8,
    chunk_size: int = 500,
    no_ch0: bool = False,
    no_ch1: bool = False,
    prefix: str = 'test',
    temp_correction: str = 'bias',
    temp_wait: int = 60,
    mock_esp: bool = False,
    mock_adc: bool = False,
    mock_pm100: bool = False,
    overwrite: bool = False,
    hwp_cycles: float = 0,
    hwp_step: float = 45,
    hwp_delay_position: float = 5,
    mc_wait: float = 15,
    disable_pm100: bool = True,
    work_dir: str = ct.WORK_DIR
):

    input_parameters = locals().copy()
    del input_parameters['work_dir']
    metadata = dict(input_parameters=input_parameters)

    logger.info("Preparing output folder...")
    params = "cycles{}-step{}-samples{}".format(cycles, step, samples)
    measurement_name = f"{date.today()}-{prefix}-{params}"
    output_folder = Path(work_dir).joinpath(ct.OUTPUT_FOLDER_DATA)
    measurement_dir = output_folder.joinpath(measurement_name)
    os.makedirs(measurement_dir, exist_ok=True)

    logger.info("Configuring log file handler...")
    log_filename = Path(output_folder).joinpath("{}.log".format(measurement_name))
    setup_log_file(log_filename)

    logger.info("Connecting to ESP Motion Controller...")
    motion_controller = ESP301.build(
        MOTION_CONTROLLER_PORT_WIN, b=MOTION_CONTROLLER_BAUDRATE,
        useaxes=[1, 2], mock_serial=mock_esp)

    logger.info("Connecting Rotary Stage: Analyzer...")
    analyzer = RotaryStage(
        motion_controller,
        cycles,
        step,
        delay_position,
        velocity,
        acceleration,
        deceleration,
        axis=1,
        name='Analyzer'
    )

    logger.info("Connecting Rotary Stage: HalfWavePlate...")
    hwp = RotaryStage(
        motion_controller, hwp_cycles, hwp_step, hwp_delay_position, axis=2, name='HalfWavePlate')

    logger.info("Connecting to ADC...")
    adc = ADC.build(
        resolve_adc_port(), baudrate=ADC_BAUDRATE, timeout=ADC_TIMEOUT,
        timeout_open=ADC_TIMEOUT_OPEN,
        ch0=not no_ch0,
        ch1=not no_ch1,
        mock_serial=mock_adc)

    pm100 = None
    if disable_pm100:
        logger.warning("Thorlabs PM100 disabled.")
    else:
        logger.info("Connecting to Thorlabs PM100...")
        try:
            pm100 = PM100.build(
                resolve_pm100_resource(),
                duration=PM100.average_count_from_duration(adc.measurement_time(samples)),
                mock=mock_pm100
            )
        except PM100Error:
            logger.warning("Thorlabs PM100 connection not found.")

    logger.info("Building DataFile...")
    data_file = DataFile(
        overwrite, header=FILE_HEADER, column_names=FILE_COLUMNS, delimiter=FILE_DELIMITER,
        output_dir=measurement_dir
    )

    logger.info("Building TemperatureFile...")
    temp_header = TEMP_HEADER.format(temp_wait)
    temp_file = DataFile(
        overwrite, header=temp_header, column_names=TEMP_COLUMNS, delimiter=FILE_DELIMITER,
        output_dir=measurement_dir
    )

    logger.info("Building Polarimeter...")
    polarimeter = Polarimeter(
        adc, analyzer, hwp, data_file, temp_file, norm_det=pm100, wait=mc_wait
    )

    logger.info("Starting measurement...")
    _, elapsed_time = timing(polarimeter.start)(samples, chunk_size=chunk_size,
                                                reps=reps, temp_wait=temp_wait,
                                                temp_correction=temp_correction)
    metadata['duration'] = str(timedelta(seconds=elapsed_time)).split(".")[0]

    logger.info("Writing measurement metadata...")
    metadata_file = measurement_dir.joinpath("metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
