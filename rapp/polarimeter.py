import os
import time
import logging

from datetime import datetime

import serial        # noqa

from rapp.esp import ESP  # noqa
from rapp.mocks import SerialMock, ESPMock
from rapp.utils import frange

from rapp.signal.analysis import plot_two_signals

logger = logging.getLogger(__name__)


OUTPUT_DIR = "output-data"

ADC_DEVICE = 'COM4'
ADC_BAUDRATE = 57600
ADC_TIMEOUT = 0.1
ADC_MAX_VAL = 4.096
ADC_MULTIPLIER_mV = 0.125

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


def bits_to_volts(value):
    return value * ADC_MULTIPLIER_mV / 1000


def read_data(adc, n_samples, in_bytes=True):
    data = []
    while len(data) < n_samples:
        try:
            if in_bytes:
                value = adc.read(2)
                value = int.from_bytes(value, byteorder='big', signed=True)
            else:
                value = adc.readline().decode().strip()
                value = int(value)

            value = bits_to_volts(value)
            data.append(value)
        except (ValueError, UnicodeDecodeError) as e:
            logger.warning(e)

    return data


def adc_acquire(adc, n_samples, **kwargs):
    adc.write(bytes(str(n_samples), 'utf-8'))
    # Sending directly the numerical value didn't work.
    # See: https://stackoverflow.com/questions/69317581/sending-serial-data-to-arduino-works-in-serial-monitor-but-not-in-python  # noqa

    a0 = read_data(adc, n_samples, **kwargs)
    a1 = read_data(adc, n_samples, **kwargs)

    return a0, a1


def main(
    cycles=1, step=10, samples=10, delay_position=1, analyzer_velocity=2, prefix='test', test=False
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename = FILENAME_FORMAT.format(prefix=prefix, cycles=cycles, step=step, samples=samples)
    filepath = os.path.join(OUTPUT_DIR, filename)
    overwrite = ask_for_overwrite(filepath)
    file = create_or_open_file(filepath, overwrite)

    if test:
        # If this happens, we don't use real connections. We use mock objects to test the code.
        # TODO: create a unit test and use the mocks from there.
        adc = SerialMock()
        analyzer = ESPMock()
    else:
        logger.info("Connecting to ADC...")
        adc = serial.Serial(ADC_DEVICE, ADC_BAUDRATE, timeout=ADC_TIMEOUT)
        logger.info("Connecting to ESP...")
        analyzer = ESP(dev=ANALYZER_DEVICE, b=ANALYZER_BAUDRATE, axis=ANALYZER_AXIS, reset=True)

    logger.info("Setting analyzer velocity to {} deg/s.".format(analyzer_velocity))
    analyzer.setvel(vel=analyzer_velocity)
    analyzer.sethomevel(vel=analyzer_velocity)

    angles = [i for i in frange(0, 360 * cycles, step)]
    logger.debug("Angles to measure: {}".format(angles))

    for angle in angles:
        analyzer.setpos(angle)      # TODO: Try to set exact position desired instead of x.001.
        time.sleep(delay_position)  # wait for position to stabilize
        adc.flushInput()            # Clear buffer. Otherwise messes up values at the beginning.

        logger.info("Angle: {}".format(analyzer.getpos()))

        a0, a1 = adc_acquire(adc, samples, in_bytes=True)

        for a0, a1 in zip(a0, a1):
            logger.debug("(A0, A1) = ({}, {})".format(a0, a1))
            datetime_ = datetime.now().isoformat()
            row = FILE_ROW.format(angle=angle, a0=a0, a1=a1, datetime=datetime_, d=FILE_DELIMITER)
            file.write(row)
            file.write('\n')

    file.close()
    adc.close()
    analyzer.dev.close()

    logger.info("Plotting result...")
    plot_two_signals(filepath, delimiter=FILE_DELIMITER, usecols=(0, 1, 2), show=True)
