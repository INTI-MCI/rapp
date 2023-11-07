import os
import time
import logging

from datetime import datetime

import serial        # noqa

from rapp.esp import ESP  # noqa
from rapp.mocks import SerialMock, ESPMock
from rapp.utils import frange

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

EXPAND_TABS = 10
FILE_HEADER = "ANGLE{d}A0{d}A1{d}DATETIME".format(d="\t").expandtabs(EXPAND_TABS)
FILE_ROW = "{angle}\t{a0}\t{a1}\t{datetime}"


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


def read_data(adc, n_samples):
    data = []
    while len(data) < n_samples:
        try:
            value = adc.readline().decode().strip()
            if value:
                value = bits_to_volts(int(value))
                data.append(value)
        except (ValueError, UnicodeDecodeError) as e:
            logger.warning(e)

    return data


def acquire(adc, n_samples):
    adc.write(bytes(str(n_samples), 'utf-8'))
    # Sending directly the numerical value didn't work.
    # See: https://stackoverflow.com/questions/69317581/sending-serial-data-to-arduino-works-in-serial-monitor-but-not-in-python  # noqa

    a0 = read_data(adc, n_samples)
    a1 = read_data(adc, n_samples)

    data = zip(a0, a1)

    return data


def main(
    cycles=1, step=10, samples=10, delay_position=1, analyzer_velocity=2, prefix='test', test=False
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if test:
        # If this happens, we don't use real connections. We use mock objects to test the code.
        # TODO: create a unit test and use the mocks from there.
        adc = SerialMock()
        analyzer = ESPMock()
    else:
        adc = serial.Serial(ADC_DEVICE, ADC_BAUDRATE, timeout=ADC_TIMEOUT)
        analyzer = ESP(dev=ANALYZER_DEVICE, b=ANALYZER_BAUDRATE, axis=ANALYZER_AXIS, reset=True)

    logger.info("Setting analyzer velocity to {} deg/s.".format(analyzer_velocity))
    analyzer.setvel(vel=analyzer_velocity)

    filename = FILENAME_FORMAT.format(prefix=prefix, cycles=cycles, step=step, samples=samples)
    filepath = os.path.join(OUTPUT_DIR, filename)
    overwrite = ask_for_overwrite(filepath)
    file = create_or_open_file(filepath, overwrite)

    angles = [i for i in frange(0, 360 * cycles, step)]
    logger.debug("Angles to measure: {}".format(angles))

    for angle in angles:
        analyzer.setpos(angle)      # TODO: Try to set exact position desired instead of x.001.
        time.sleep(delay_position)  # wait for position to stabilize
        adc.flushInput()            # Clear buffer. Otherwise messes up values at the beginning.

        logger.info("Angle: {}".format(analyzer.getpos()))

        data = acquire(adc, samples)

        for a0, a1 in data:
            logger.debug("(A0, A1) = ({}, {})".format(a0, a1))
            datetime_ = datetime.now().isoformat()
            row = FILE_ROW.format(angle=angle, a0=a0, a1=a1, datetime=datetime_)
            file.write(row.expandtabs(EXPAND_TABS))
            file.write('\n')

    file.close()
    adc.close()
    analyzer.dev.close()
