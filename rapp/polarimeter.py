import os
import time
import logging


from datetime import datetime

import serial        # noqa
from rapp.esp import ESP  # noqa
from rapp.mocks import SerialMock, ESPMock


logger = logging.getLogger(__name__)

FILE_HEADER = "ANGLE [°], A0 [V], A1 [V], DATETIME"
FILE_ROW = "{angle} {a0} {a1} {datetime}"

OUTPUT_DIR = "output"

ADC_DEVICE = 'COM4'
ADC_BAUDRATE = 57600
ADC_TIMEOUT = 0.1
ADC_MAX_VAL = 4.096

ANALYZER_DEVICE = "COM3"
ANALYZER_BAUDRATE = 921600
ANALYZER_AXIS = 1


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
        if i == 's':
            overwrite = True

    return overwrite


def parse_data(data):
    if not data:
        raise ValueError("Data is empty!")

    data_list = data.split(",")
    if len(data_list) != 2:
        raise ValueError("Expected 2 values! Got {}.".format(len(data_list)))

    a0, a1 = data_list

    if (
        not (-ADC_MAX_VAL <= float(a0) <= ADC_MAX_VAL) or
        not (-ADC_MAX_VAL <= float(a1) <= ADC_MAX_VAL)
    ):
        raise ValueError('Values out of the range [-{}, {}]'.format(ADC_MAX_VAL, ADC_MAX_VAL))

    return a0, a1


def frange(start, end, step):
    """A range with float step allowed."""
    return [p * step for p in range(start, int(end / step))]


def main(
    cycles=1, step=10, samples=10, delay_position=1, delay_angle=0, analyzer_velocity=2,
    filename='test', test=False
):

    filename = "{}-cycles{}-step{}-samples{}.txt".format(filename, cycles, step, samples)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    overwrite = ask_for_overwrite(filepath)

    angles = [i for i in frange(0, 360 * cycles, step)]

    logger.debug("Angles to measure: {}".format(angles))

    if test:
        # If this happens, we don't use real connections. We use mock objects to test the code.
        # TODO: create a unit test and use the mocks from there.
        serialport = SerialMock()
        analyzer = ESPMock()
    else:
        serialport = serial.Serial(ADC_DEVICE, ADC_BAUDRATE, timeout=ADC_TIMEOUT)
        analyzer = ESP(dev=ANALYZER_DEVICE, b=ANALYZER_BAUDRATE, axis=ANALYZER_AXIS, reset=True)

    analyzer.setvel(vel=analyzer_velocity)

    file = create_or_open_file(filepath, overwrite)
    for angle in angles:
        analyzer.setpos(angle)   # TODO: Try to obtain the exact position desired instead of .0001.
        time.sleep(delay_position)  # wait for position to stabilize
        serialport.flushInput()  # Clear buffer. Otherwise messes up measurements at the beginning.

        logger.info(f"Measuring at position: {analyzer.getpos()}")

        i = 0
        while i < samples:
            data = serialport.readline().decode().strip()

            try:
                a0, a1 = parse_data(data)
            except ValueError as e:
                logger.warning("Found error in data: {}. data: {}. Skipping...".format(e, data))
                continue

            datetime_ = datetime.now().isoformat()
            row = FILE_ROW.format(angle=angle, a0=a0, a1=a1, datetime=datetime_)

            logger.debug(row)

            file.write(row)
            file.write('\n')
            i = i + 1

            time.sleep(delay_angle)

    file.close()
    serialport.close()
    analyzer.dev.close()
