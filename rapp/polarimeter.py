import os
import sys
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


ADC_WIN_DEVICE = 'COM4'
ADC_LINUX_DEVICE = '/dev/ttyACM0'

ADC_BAUDRATE = 57600
ADC_TIMEOUT = 0.1
ADC_WAIT = 2

ANALYZER_DEVICE = "COM3"
ANALYZER_BAUDRATE = 921600
ANALYZER_AXIS = 1

FILE_NAME = "{prefix}-cycles{cycles}-step{step}-samples{samples}.txt"
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


def resolve_adc_device():
    if sys.platform == 'linux':
        return ADC_LINUX_DEVICE

    return ADC_WIN_DEVICE


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
    cycles=1, step=10, samples=10, delay_position=1, velocity=2, no_ch0=False, no_ch1=False,
    chunk_size=2000, prefix='test', test_esp=False, test_adc=False, plot=False
):
    output_dir = os.path.join(ct.WORK_DIR, ct.OUTPUT_FOLDER_DATA)
    os.makedirs(output_dir, exist_ok=True)

    if test_esp:
        logger.warning("Using ESP mock object.")
        analyzer = ESPMock()
    else:
        logger.info("Connecting to ESP...")
        analyzer = ESP(ANALYZER_DEVICE, b=ANALYZER_BAUDRATE, axis=ANALYZER_AXIS)

    if test_adc:
        logger.warning("Using ADC mock object.")
        adc = ADCMock()
    else:
        logger.info("Connecting to ADC...")
        adc = ADC(resolve_adc_device(), b=ADC_BAUDRATE, timeout=ADC_TIMEOUT, wait=ADC_WAIT)

    logger.info("Setting analyzer velocity to {} deg/s.".format(velocity))
    analyzer.setvel(vel=velocity)
    analyzer.setvel(vel=velocity, axis=2)

    logger.info("Setting analyzer home velocity to {} deg/s.".format(velocity))
    analyzer.sethomevel(vel=velocity)
    analyzer.sethomevel(vel=velocity, axis=2)

    logger.info("Samples to measure in each angle: {}.".format(samples))
    logger.info("Maximum chunk size: {}.".format(chunk_size))

    if chunk_size > 0:
        chunks = get_chunks(range(samples), chunk_size)
        chunks_sizes = [len(x) for x in chunks]
    else:
        chunks_sizes = [samples]

    for hwp_angle in [0, 4.5]:
        for rep in range(1, 6):
            prefix_new = "{}-hwp{}-rep{}".format(prefix, hwp_angle, rep)
            filename = FILE_NAME.format(prefix=prefix_new, cycles=cycles, step=step, samples=samples)
            filepath = os.path.join(output_dir, filename)
            overwrite = ask_for_overwrite(filepath)
            file = open_file(filepath, overwrite)

            today = date.today()
            file_meta = FILE_METADATA.format(d=today, cycles=cycles, step=step, samples=samples)
            file_header = "{}{}\n".format(file_meta, FILE_COLUMNS)
            file.write(file_header)

            analyzer.setpos(hwp_angle, axis=2)
            logger.info("Waiting 5 seconds after changing half wave plate position...")
            time.sleep(5)

            init_position = analyzer.getpos()
            logger.info("Analyzer current position: {}".format(init_position))

            if cycles == 0:
                angles = [init_position]
                adc.progressbar = True
            else:
                angles = generate_angles(cycles, step, init_position=init_position)
                adc.progressbar = False

            logger.info("Will measure {} angles: {}.".format(len(angles), angles))

            for angle in progressbar(angles, prefix="Angles:", enable=len(angles) > 1):
                logger.debug("Changing analyzer position...")
                analyzer.setpos(angle)

                logger.debug("Waiting {}s after changing position...".format(delay_position))
                time.sleep(delay_position)

                for chunk_size in chunks_sizes:
                    adc.flush_input()  # Clear buffer. Otherwise messes up values at the beginning.
                    data = adc.acquire(chunk_size, ch0=not no_ch0, ch1=not no_ch1, in_bytes=True)

                    channels_names_tuple = "({})".format(", ".join(data.keys()))
                    ch0, ch1 = data.values()
                    for i in range(len(ch0)):
                        logger.debug("{} = ({}, {})".format(channels_names_tuple, ch0[i], ch1[i]))
                        row = FILE_ROW.format(
                            angle=angle,
                            ch0=ch0[i],
                            ch1=ch1[i],
                            datetime=datetime.now().isoformat(),
                            s=FILE_DELIMITER
                        )
                        file.write(row.expandtabs(10))
                        file.write('\n')

    file.close()
    adc.close()
    analyzer.close()

    logger.info("Done!")

    if plot:
        logger.info("Plotting result...")
        plot_two_signals(filepath, output_dir, sep=r"\s+", show=True)
