import os
import time
import logging

from datetime import datetime

import numpy as np


from rapp import constants as ct

from rapp.esp import ESP
from rapp.adc import ADC
from rapp.utils import frange
from rapp.mocks import ADCMock, ESPMock
from rapp.signal.analysis import plot_two_signals

np.set_printoptions(threshold=0, edgeitems=5)  # truncate arrays when printing
logger = logging.getLogger(__name__)


ADC_DEVICE = 'COM4'
ADC_BAUDRATE = 57600
ADC_TIMEOUT = 0.1
ADC_WAIT = 2

ANALYZER_DEVICE = "COM3"
ANALYZER_BAUDRATE = 921600
ANALYZER_AXIS = 1

MAX_CHUNK_SIZE = 500

FILENAME_FORMAT = "{prefix}-cycles{cycles}-step{step}-samples{samples}.txt"

FILE_DELIMITER = "\t"
FILE_HEADER = "ANGLE{d}A0{d}A1{d}DATETIME".format(d=FILE_DELIMITER)
FILE_ROW = "{angle}{d}{a0}{d}{a1}{d}{datetime}"


def create_file(filename):
    file = open(filename, 'w')
    file.write(FILE_HEADER)
    file.write('\n')
    return file


def open_file(filename, overwrite=False):
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


def get_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main(
    cycles=1, step=10, samples=10, delay_position=1, analyzer_velocity=2, prefix='test',
    test=False, plot=False
):
    output_dir = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_DATA)
    os.makedirs(output_dir, exist_ok=True)

    filename = FILENAME_FORMAT.format(prefix=prefix, cycles=cycles, step=step, samples=samples)
    filepath = os.path.join(output_dir, filename)
    overwrite = ask_for_overwrite(filepath)
    file = open_file(filepath, overwrite)

    if test:
        # If this happens, we don't use real connections. We use mock objects to test the code.
        # TODO: create a unit test and use the mocks from there.
        adc = ADCMock()
        analyzer = ESPMock()
    else:
        logger.info("Connecting to ADC...")
        adc = ADC(ADC_DEVICE, b=ADC_BAUDRATE, timeout=ADC_TIMEOUT, wait=ADC_WAIT)

        logger.info("Connecting to ESP...")
        analyzer = ESP(ANALYZER_DEVICE, b=ANALYZER_BAUDRATE, axis=ANALYZER_AXIS)

    logger.info("Setting analyzer velocity to {} deg/s.".format(analyzer_velocity))
    analyzer.setvel(vel=analyzer_velocity)

    logger.info("Setting analyzer home velocity to {} deg/s.".format(analyzer_velocity))
    analyzer.sethomevel(vel=analyzer_velocity)

    logger.info("Samples to measure in each angle: {}.".format(samples))
    logger.info("Maximum chunk size configured: {}.".format(MAX_CHUNK_SIZE))

    chunks = get_chunks(range(samples), MAX_CHUNK_SIZE)
    chunks_sizes = [len(x) for x in chunks]
    logger.info("Samples chunks sizes: {}".format(chunks_sizes))

    init_position = analyzer.getpos()

    if cycles == 0:
        angles = [init_position]
    else:
        angles = np.array([i for i in frange(init_position, init_position + 360 * cycles, step)])

    logger.info("Angles to process: {}.".format(angles))
    for angle in angles:
        logger.debug("Changing analyzer position...")
        analyzer.setpos(angle)

        logger.debug("Waiting {}s after changing position...".format(delay_position))
        time.sleep(delay_position)

        logger.info("Measuring angle {}°".format(angle))
        for chunk_size in chunks_sizes:
            adc.flush_input()  # Clear buffer. Otherwise messes up values at the beginning.
            data = adc.acquire(chunk_size, in_bytes=True)

            for a0, a1 in data:
                logger.debug("(A0, A1) = ({}, {})".format(a0, a1))
                dt = datetime.now().isoformat()
                row = FILE_ROW.format(angle=angle, a0=a0, a1=a1, datetime=dt, d=FILE_DELIMITER)
                file.write(row)
                file.write('\n')

    file.close()
    adc.close()
    analyzer.dev.close()

    if plot:
        logger.info("Plotting result...")
        plot_two_signals(filepath, output_dir, delimiter=FILE_DELIMITER, show=True)
