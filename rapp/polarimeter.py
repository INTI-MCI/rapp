import os
import time
import logging

from datetime import datetime

from rapp import constants as ct

from rapp.esp import ESP
from rapp.adc import ADC
from rapp.utils import frange
from rapp.mocks import ADCMock, ESPMock
from rapp.signal.analysis import plot_two_signals

logger = logging.getLogger(__name__)


ADC_DEVICE = 'COM4'
ADC_BAUDRATE = 57600
ADC_TIMEOUT = 0.1
ADC_WAIT = 2

ANALYZER_DEVICE = "COM3"
ANALYZER_BAUDRATE = 921600
ANALYZER_AXIS = 1

FILENAME_FORMAT = "{prefix}-cycles{cycles}-step{step}-samples{samples}.txt"

FILE_DELIMITER = "\t"
FILE_HEADER = "ANGLE{d}A0{d}A1{d}DATETIME".format(d=FILE_DELIMITER)
FILE_ROW = "{angle}{d}{a0}{d}{a1}{d}{datetime}"


def create_file(filename):
    file = open(filename, 'w')
    file.write(FILE_HEADER)
    file.write('\n')
    return file


def create_or_open_file(filename, overwrite):
    if not os.path.exists(filename) or overwrite:
        file = create_file(filename)
    else:
        file = open(filename, 'a')

    return file


def ask_for_overwrite(filename):
    overwrite = False
    if os.path.exists(filename):
        i = input(
            "File already exists. Do you want to erase it? y/n (default is NO): ")
        if i == 'y':
            overwrite = True

    return overwrite


def main(
    cycles=1, step=10, samples=10, delay_position=1, analyzer_velocity=2, prefix='test',
    test=False, plot=False
):
    output_dir = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_DATA)
    os.makedirs(output_dir, exist_ok=True)

    filename = FILENAME_FORMAT.format(prefix=prefix, cycles=cycles, step=step, samples=samples)
    filepath = os.path.join(output_dir, filename)
    overwrite = ask_for_overwrite(filepath)
    file = create_or_open_file(filepath, overwrite)

    if test:
        # If this happens, we don't use real connections. We use mock objects to test the code.
        # TODO: create a unit test and use the mocks from there.
        adc = ADCMock()
        analyzer = ESPMock()
    else:
        logger.info("Connecting to ADC...")
        adc = ADC(ADC_DEVICE, b=ADC_BAUDRATE, timeout=ADC_TIMEOUT, wait=ADC_WAIT)

        logger.info("Connecting to ESP...")
        analyzer = ESP(ANALYZER_DEVICE, b=ANALYZER_BAUDRATE, axis=ANALYZER_AXIS, reset=True)

    logger.info("Setting analyzer velocity to {} deg/s.".format(analyzer_velocity))
    analyzer.setvel(vel=analyzer_velocity)
    analyzer.sethomevel(vel=analyzer_velocity)

    angles = [i for i in frange(0, 360 * cycles, step)]
    logger.debug("Angles to measure: {}".format(angles))

    for angle in angles:
        analyzer.setpos(angle)      # TODO: Try to set exact position desired instead of x.001.
        time.sleep(delay_position)  # wait for position to stabilize
        adc.flush_input()            # Clear buffer. Otherwise messes up values at the beginning.

        logger.info("Angle: {}".format(analyzer.getpos()))

        data = adc.acquire(samples, in_bytes=True)

        for a0, a1 in data:
            logger.debug("(A0, A1) = ({}, {})".format(a0, a1))
            datetime_ = datetime.now().isoformat()
            row = FILE_ROW.format(angle=angle, a0=a0, a1=a1, datetime=datetime_, d=FILE_DELIMITER)
            file.write(row)
            file.write('\n')

    file.close()
    adc.close()
    analyzer.dev.close()

    if plot:
        logger.info("Plotting result...")
        plot_two_signals(filepath, delimiter=FILE_DELIMITER, usecols=(0, 1, 2), show=True)
