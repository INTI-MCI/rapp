import os
import time
import logging

from datetime import datetime, date

import numpy as np
from rapp import constants as ct

from rapp.esp import ESP
from rapp.adc import ADC
from rapp.mocks import ADCMock, ESPMock
from rapp.signal.analysis import plot_two_signals
from rapp.utils import progressbar

# Always truncate arrays when printing, without scientific notation.
np.set_printoptions(threshold=0, edgeitems=5, suppress=True)
logger = logging.getLogger(__name__)


ADC_DEVICE = 'COM4'
ADC_BAUDRATE = 57600
ADC_TIMEOUT = 0.1
ADC_WAIT = 2

ANALYZER_DEVICE = "COM3"
ANALYZER_BAUDRATE = 921600
ANALYZER_AXIS = 1

MAX_CHUNK_SIZE = 500

FILE_NAME = "{d}-{prefix}-cycles{cycles}-step{step}-samples{samples}.txt"
FILE_ROW = "{angle}{s}{ch0}{s}{ch1}{s}{datetime}"
FILE_DELIMITER = "\t"
FILE_COLUMNS = "ANGLE{s}CH0{s}CH1{s}DATETIME".format(s=FILE_DELIMITER).expandtabs(10)
FILE_METADATA = (
    "#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#\n"
    "#~~~~~~~~~ RAPP measurements | INTI {d} ~~~~~~~~#\n"
    "#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#\n"
    "# cycles  : {cycles}\n"
    "# step    : {step}\n"
    "# samples : {samples}\n"
    "# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#\n"
)


def open_file(filename, overwrite=False):
    if not os.path.exists(filename) or overwrite:
        file = open(filename, 'w')
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


def generate_angles(cycles, step, init_position=0.0):
    return np.arange(init_position, init_position + 360 * cycles + step, step)


def main(
    cycles=1, step=10, samples=10, chunks=0, delay_position=1, velocity=2, prefix='test',
    test=False, plot=False
):
    output_dir = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_DATA)
    os.makedirs(output_dir, exist_ok=True)

    today = date.today()
    filename = FILE_NAME.format(d=today, prefix=prefix, cycles=cycles, step=step, samples=samples)
    filepath = os.path.join(output_dir, filename)
    overwrite = ask_for_overwrite(filepath)
    file = open_file(filepath, overwrite)

    file_meta = FILE_METADATA.format(d=today, cycles=cycles, step=step, samples=samples)
    file_header = "{}{}\n".format(file_meta, FILE_COLUMNS)
    file.write(file_header)

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

    logger.info("Setting analyzer velocity to {} deg/s.".format(velocity))
    analyzer.setvel(vel=velocity)

    logger.info("Setting analyzer home velocity to {} deg/s.".format(velocity))
    analyzer.sethomevel(vel=velocity)

    logger.info("Samples to measure in each angle: {}.".format(samples))
    logger.info("Maximum chunk size configured: {}.".format(MAX_CHUNK_SIZE))

    if chunks != 0:
        chunks = get_chunks(range(samples), MAX_CHUNK_SIZE)
        chunks_sizes = [len(x) for x in chunks]
    else:
        chunks_sizes = [samples]

    init_position = analyzer.getpos()
    logger.info("Analyzer current position: {}".format(init_position))

    if cycles == 0:
        angles = [init_position]
    else:
        angles = generate_angles(cycles, step, init_position=init_position)

    logger.info("Will measure {} angles: {}.".format(len(angles), angles))


    # for angle in progressbar(angles, prefix="Measuring: ", size=100):
    for angle in angles:

        logger.debug("Changing analyzer position...")
        analyzer.setpos(angle)

        logger.debug("Waiting {}s after changing position...".format(delay_position))
        time.sleep(delay_position)

        for chunk_size in chunks_sizes:
            adc.flush_input()  # Clear buffer. Otherwise messes up values at the beginning.
            data = adc.acquire(chunk_size, in_bytes=True)

            for ch0, ch1 in data:
                logger.debug("(CH0, CH1) = ({}, {})".format(ch0, ch1))
                dt = datetime.now().isoformat()
                row = FILE_ROW.format(angle=angle, ch0=ch0, ch1=ch1, datetime=dt, s=FILE_DELIMITER)
                file.write(row.expandtabs(10))
                file.write('\n')

    file.close()
    adc.close()
    analyzer.dev.close()

    logger.info("Done!")

    if plot:
        logger.info("Plotting result...")
        plot_two_signals(filepath, output_dir, sep=r"\s+", show=True)
