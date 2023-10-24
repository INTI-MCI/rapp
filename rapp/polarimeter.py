import os
import sys
import time

from datetime import datetime

import serial


MEASUREMENT_MODE = 'new_file'
path = 'test.txt'

FILE_NAME = 'test.txt'
FILE_HEADER = "ANGLE [°], A0 [V], A1 [V], DATETIME"
FILE_ROW = "{angle}\t{a0}\t{a1}\t{datetime}"

ANGLES_TO_MEASURE = [0]

SERIAL_DEVICE = 'COM3'
SERIAL_BAUDRATE = 57600
SERIAL_TIMEOUT = 0.1


class SerialMock:
    """Mock object to test the code."""
    def readline(self):
        return b'3.4567,2.3422'

    def close(self):
        pass


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


def main(angles, n_points=10, delay=0, filename='test.txt', overwrite=False, verbose=False):
    # serialport = SerialMock()  # this is just to test when we don't have the device connected.
    serialport = serial.Serial(SERIAL_DEVICE, SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT)

    file = create_or_open_file(filename, overwrite)

    for angle in angles:
        # TODO: move analyzer to the angle.
        i = 0
        while i < n_points:
            data_raw = serialport.readline().decode().strip()

            if data_raw:  # TODO: Check if this IF is really needed now that we removed the delay.
                a0, a1 = data_raw.split(",")
                datetime_ = datetime.now().isoformat()
                row = FILE_ROW.format(angle=angle, a0=a0, a1=a1, datetime=datetime_)

                if verbose:
                    print(row)

                file.write(row)
                file.write('\n')
                i = i + 1

                time.sleep(delay)

    file.close()
    serialport.close()


if __name__ == '__main__':
    n_points = int(sys.argv[1])          # pass how many points to measure.
    delay = float(sys.argv[2])           # pass the delay (in seconds) between measurements.
    filename = str(sys.argv[3])          # pass the filename.
    overwrite = bool(int(sys.argv[4]))   # pass 1 (true) or 0 (false).
    verbose = bool(int(sys.argv[5]))     # pass 1 (true) or 0 (false).

    # example:
    # python polarimeter.py 10 0 test.txt 1 0

    main(ANGLES_TO_MEASURE, n_points, delay, filename, overwrite, verbose)
